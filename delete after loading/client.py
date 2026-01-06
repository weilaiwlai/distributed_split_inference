import torch
import requests
import json
import time
import uuid
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from typing import Optional
import gc
import os
from data import get_random_prompted_examples
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

class Config:
    def __init__(self):
        self.subset = "main"
        self.split = "test"
        self.num_examples = 20
        self.random_seed = 42

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def get_device_map(
    client_attention_layers=4,
    total_layers=32,
    client=True,
    tail=False,
    server_gpus=None
):
    device_map = {}
    if server_gpus is None:
        server_gpus = [0, 1] 

    device_map["model.embed_tokens.weight"] = 0 if (client and not tail) else server_gpus[-1]
    device_map["model.norm.weight"] = server_gpus[-1]
    device_map["lm_head.weight"] = 0 if (client and tail) else server_gpus[-1]

    for i in range(total_layers):
        prefix = f"model.layers.{i}"
        attn = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"

        if client and not tail:
            device = 0 if i < client_attention_layers else "disk"
        elif client and tail:
            device = "disk"
        else:
            if i < client_attention_layers:
                device = "disk"
            else:
                server_layer_idx = i - client_attention_layers
                n_server_layers = total_layers - client_attention_layers
                n_gpus = len(server_gpus)
                gpu_index = int(server_layer_idx / n_server_layers * n_gpus)
                gpu_index = min(gpu_index, n_gpus - 1)
                device = server_gpus[gpu_index]

        # Attention
        for w in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            device_map[f"{attn}.{w}.weight"] = device
        # MLP
        for w in ["gate_proj", "up_proj", "down_proj"]:
            device_map[f"{mlp}.{w}.weight"] = device
        # Norms
        device_map[f"{prefix}.input_layernorm.weight"] = device
        device_map[f"{prefix}.post_attention_layernorm.weight"] = device
        # Rotary
        device_map[f"{attn}.rotary_emb.inv_freq"] = device

    return device_map

class ClientHead:
    def __init__(self, model_name, client_attention_layers, total_layers):
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=True,
            tail=False,
            server_gpus=[0, 1]
        )
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        # Keep only the client layers
        self.base.model.layers = self.base.model.layers[:client_attention_layers]
        # Disable unused parts
        self.base.lm_head = torch.nn.Identity()
        self.base.model.norm = torch.nn.Identity()

        # Session storage for client-side KV cache (per session_id)
        self.client_sessions = {}

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, session_id: str = "default"):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Retrieve or create client-side cache
        if session_id in self.client_sessions:
            past_cache = self.client_sessions[session_id]
            past_len = past_cache.get_seq_length()
        else:
            past_cache = DynamicCache()
            past_len = 0

        total_len = past_len + seq_len
        position_ids = torch.arange(past_len, total_len, device=device).unsqueeze(0)  # [1, seq_len]
        cache_position = torch.arange(past_len, total_len, device=device)             # [seq_len]

        inputs_embeds = self.base.model.embed_tokens(input_ids)  # [B, L, D]
        hidden_states = inputs_embeds

        # Build causal mask for full sequence (past + current)
        # But note: we only have current embeds, so use standard 4D mask with past_len
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        full_attention_mask = torch.ones((batch_size, total_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=full_attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_len,
            sliding_window=getattr(self.base.config, "sliding_window", None),
        )

        # Compute RoPE embeddings
        cos, sin = self.base.model.rotary_emb(inputs_embeds, position_ids)

        # Forward through client layers
        for layer in self.base.model.layers:
            layer_device = layer.input_layernorm.weight.device
            layer_hidden = hidden_states.to(layer_device)
            layer_position_ids = position_ids.to(layer_device)
            layer_attention_mask = attention_mask.to(layer_device)
            layer_cos = cos.to(layer_device)
            layer_sin = sin.to(layer_device)
            layer_cache_position = cache_position.to(layer_device)

            layer_output = layer(
                hidden_states=layer_hidden,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                position_embeddings=(layer_cos, layer_sin),
                past_key_values=past_cache,      # ← pass cache
                use_cache=True,
                cache_position=layer_cache_position,
            )
            hidden_states = layer_output  # only hidden_states returned

        # Save updated cache
        self.client_sessions[session_id] = past_cache
        return hidden_states

    def clear_session(self, session_id: str):
        self.client_sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        return session_id in self.client_sessions

class ClientTail:
    def __init__(self, model_name, client_attention_layers, total_layers):
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=True,
            tail=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        self.lm_head = base.lm_head
        del base

    @torch.no_grad()
    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

def warmup_client_components(client_head: ClientHead, client_tail: ClientTail, server_url: str, n_warmup_runs=2):
    print("\n=== 开始客户端组件预热 ===")
    
    # 创建模拟的输入用于预热
    dummy_input_ids = torch.randint(0, 1000, (1, 100), dtype=torch.long, device='cuda:0')
    
    for i in range(n_warmup_runs):
        print(f"正在进行第 {i+1}/{n_warmup_runs} 次客户端预热...")
        
        # 预热 ClientHead
        start_time = time.time()
        head_out = client_head.forward(dummy_input_ids, session_id=f"client_warmup_{i}")
        head_time = time.time() - start_time
        
        print(f"ClientHead 预热运行 {i+1} 完成，耗时: {head_time:.2f} 秒")
        print(f"ClientHead 输出形状: {head_out.shape}")
        
        # 预热 ClientTail
        start_time = time.time()
        tail_out = client_tail.forward(head_out)
        tail_time = time.time() - start_time
        
        print(f"ClientTail 预热运行 {i+1} 完成，耗时: {tail_time:.2f} 秒")
        print(f"ClientTail 输出形状: {tail_out.shape}")
        
        # 预热服务器通信
        try:
            hidden_states_list = head_out.to(torch.float32).cpu().numpy().tolist()
            
            response = requests.post(
                f"http://{server_url}/process_hidden_states",
                json={
                    "hidden_states": hidden_states_list,
                    "session_id": f"server_warmup_{i}"
                }
            )
            
            if response.status_code == 200:
                print(f"服务器通信检查运行 {i+1} 成功")
            else:
                print(f"服务器通信检查运行 {i+1} 失败: {response.status_code}")
        except Exception as e:
            print(f"服务器通信检查运行 {i+1} 出错: {str(e)}")
    
    print("=== 客户端组件预热完成 ===")

class DistributedLLMGenerator:
    def __init__(self, model_name, client_head, client_tail, server_url):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.client_head = client_head
        self.client_tail = client_tail
        self.server_url = server_url
        self.SYSTEM_PROMPT = "You are a helpful AI assistant."

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_k: Optional[int] = 50,
        top_p: float = 0.9,
        session_id: Optional[str] = None,
    ):
        if session_id is None:
            session_id = str(uuid.uuid4())

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to('cuda:0')
        total_len = input_ids.shape[1]

        # Process through client head
        head_out = self.client_head.forward(input_ids, session_id=session_id)
        
        # Send to server for backend processing
        # Convert bfloat16 to float32 before converting to numpy
        hidden_states_list = head_out.to(torch.float32).cpu().numpy().tolist()
        
        response = requests.post(
            f"http://{self.server_url}/process_hidden_states",
            json={
                "hidden_states": hidden_states_list,
                "session_id": session_id
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Server request failed: {response.status_code}, {response.text}")
        
        server_output_data = response.json()
        # Convert the response back to bfloat16 tensor
        backend_out = torch.tensor(server_output_data["hidden_states"], dtype=torch.float32).to('cuda:0').bfloat16()
        
        # Process through client tail to get logits
        logits = self.client_tail.forward(backend_out)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k:
            k = min(top_k, next_token_logits.size(-1))
            top_k_vals, top_k_idx = torch.topk(next_token_logits, k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits.scatter_(1, top_k_idx, top_k_vals)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = 0
            indices_to_remove = sorted_indices[remove_mask]
            next_token_logits[:, indices_to_remove] = float('-inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        total_len += 1

        for _ in range(1, max_new_tokens):
            new_token_id = next_token
            head_out = self.client_head.forward(new_token_id, session_id=session_id)
            
            # Send to server for backend processing
            # Convert bfloat16 to float32 before converting to numpy
            hidden_states_list = head_out.to(torch.float32).cpu().numpy().tolist()
            
            response = requests.post(
                f"http://{self.server_url}/process_hidden_states",
                json={
                    "hidden_states": hidden_states_list,
                    "session_id": session_id
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Server request failed: {response.status_code}, {response.text}")
            
            server_output_data = response.json()
            # Convert the response back to bfloat16 tensor
            backend_out = torch.tensor(server_output_data["hidden_states"], dtype=torch.float32).to('cuda:0').bfloat16()
            
            # Process through client tail to get logits
            logits = self.client_tail.forward(backend_out)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k:
                k = min(top_k, next_token_logits.size(-1))
                top_k_vals, top_k_idx = torch.topk(next_token_logits, k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_idx, top_k_vals)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = 0
                indices_to_remove = sorted_indices[remove_mask]
                next_token_logits[:, indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            total_len += 1

            if next_token.item() in [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]:
                break

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        total_steps = total_len - input_ids.shape[1]
        print(f"\n[Timing] Generated {total_steps} tokens")
        return response, total_steps

if __name__ == "__main__":
    # Configuration
    model_name = "/opt/models/Qwen3-32B"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    client_attention_layers = 1
    total_layers = config.num_hidden_layers

    # Initialize client components
    print(f"Loading client components...")
    client_head = ClientHead(model_name, client_attention_layers, total_layers)
    client_tail = ClientTail(model_name, client_attention_layers, total_layers)

    # Initialize distributed generator with server URL
    server_url = "127.0.0.1:8000"  # Replace with your actual server IP and port
    generator = DistributedLLMGenerator(model_name, client_head, client_tail, server_url)

    warmup_client_components(client_head, client_tail, server_url)

    #Warmup
    prompt = "Warmup test"
    start = time.time()
    response = generator.generate(prompt, max_new_tokens=256, temperature=0.7)
    elapsed = time.time() - start
    print("\nResponse:\n", response)

    dataset = "gsm8k"  #cnn_dailymail mbpp openai_humaneval alpaca gsm8k
    config = Config()
    examples = get_random_prompted_examples(dataset, config)
    for i, example in enumerate(examples):
        print(f"\n=== Processing example {i+1}/{len(examples)} ===")
        start = time.time()
        response, total_tokens = generator.generate(example, max_new_tokens=256, temperature=0.7)
        elapsed = time.time() - start
        print("\nResponse:\n", response)
        print(f"\nTime: {elapsed:.2f}s | Speed: {total_tokens / elapsed:.2f} tokens/s")