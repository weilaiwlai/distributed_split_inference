from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional
import os
import gc
import time
import uuid
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import DynamicCache
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

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
        server_gpus = [1,2] 

    # --- Embedding, final norm, lm_head ---
    device_map["model.embed_tokens.weight"] = 0 if (client and not tail) else server_gpus[-1]
    device_map["model.norm.weight"] = server_gpus[-1]  # 最终 norm 放最后一张卡
    device_map["lm_head.weight"] = 0 if (client and tail) else server_gpus[-1]

    # --- All layers: 0 to total_layers - 1 ---
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
                gpu_index = min(gpu_index, n_gpus - 1)  # 防越界
                device = server_gpus[gpu_index]

        # === Attention Projections ===
        device_map[f"{attn}.q_proj.weight"] = device
        device_map[f"{attn}.q_proj.bias"] = device
        device_map[f"{attn}.k_proj.weight"] = device
        device_map[f"{attn}.k_proj.bias"] = device
        device_map[f"{attn}.v_proj.weight"] = device
        device_map[f"{attn}.v_proj.bias"] = device
        device_map[f"{attn}.o_proj.weight"] = device

        # === Qwen-specific: q_norm & k_norm ===
        if f"{attn}.q_norm.weight" in device_map:
            device_map[f"{attn}.q_norm.weight"] = device
        if f"{attn}.k_norm.weight" in device_map:
            device_map[f"{attn}.k_norm.weight"] = device

        # === MLP ===
        device_map[f"{mlp}.gate_proj.weight"] = device
        device_map[f"{mlp}.up_proj.weight"] = device
        device_map[f"{mlp}.down_proj.weight"] = device

        # === LayerNorms ===
        device_map[f"{prefix}.input_layernorm.weight"] = device
        device_map[f"{prefix}.post_attention_layernorm.weight"] = device

        # === 偏置项 (如果存在) ===
        if f"{attn}.q_proj.bias" in device_map:
            device_map[f"{attn}.q_proj.bias"] = device
        if f"{attn}.k_proj.bias" in device_map:
            device_map[f"{attn}.k_proj.bias"] = device
        if f"{attn}.v_proj.bias" in device_map:
            device_map[f"{attn}.v_proj.bias"] = device
        if f"{mlp}.gate_proj.bias" in device_map:
            device_map[f"{mlp}.gate_proj.bias"] = device
        if f"{mlp}.up_proj.bias" in device_map:
            device_map[f"{mlp}.up_proj.bias"] = device
        if f"{mlp}.down_proj.bias" in device_map:
            device_map[f"{mlp}.down_proj.bias"] = device

        # === Rotary Embedding buffer (safe to include) ===
        device_map[f"{attn}.rotary_emb.inv_freq"] = device

    return device_map

class ClientHead:
    def __init__(self, model_name, client_attention_layers, total_layers):
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=True,
            tail=False,
        )
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/home/yueshuaibing/models/Qwen-2.5-7B',
            cache_dir='/home/yueshuaibing/models/Qwen-2.5-7B',
            trust_remote_code=True
        )
        self.base.model.layers = self.base.model.layers[:client_attention_layers]
        self.base.lm_head = torch.nn.Identity()
        self.base.model.norm = torch.nn.Identity()
        self.client_sessions = {}
        if hasattr(self.base.model, 'rotary_emb') and hasattr(self.base.model.rotary_emb, 'inv_freq'):
            self.base.model.rotary_emb.inv_freq = self.base.model.rotary_emb.inv_freq.to("cuda:0")

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, session_id: str = "default"):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

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

        full_attention_mask = torch.ones((batch_size, total_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=full_attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_len,
            sliding_window=getattr(self.base.config, "sliding_window", None),
        )

        position_embeddings = self.base.model.rotary_emb(inputs_embeds, position_ids)

        # Forward through client layers
        for layer in self.base.model.layers:
            #t0 = time.time()
            layer_output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_cache,   
                use_cache=True,
                cache_position=cache_position,
            )
            #print(time.time() - t0)
            hidden_states = layer_output 
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
            offload_folder='/home/yueshuaibing/models/Qwen-2.5-7B',
            cache_dir='/home/yueshuaibing/models/Qwen-2.5-7B',
            trust_remote_code=True
        )
        self.lm_head = base.lm_head
        del base

    @torch.no_grad()
    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

class ModelBackend:
    def __init__(self, model_name, client_attention_layers, total_layers):
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=False,
            tail=False,
            server_gpus=[1,2]
        )
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/home/yueshuaibing/models/Qwen-2.5-7B',
            cache_dir='/home/yueshuaibing/models/Qwen-2.5-7B',
            trust_remote_code=True
        )
        self.base.model.layers = self.base.model.layers[client_attention_layers:]
        self.base.lm_head = torch.nn.Identity()
        
        norm_device = self.base.model.norm.weight.device
        if hasattr(self.base.model.rotary_emb, 'inv_freq'):
            self.base.model.rotary_emb.inv_freq = self.base.model.rotary_emb.inv_freq.to(norm_device)
        self.sessions = {}

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, session_id: str = "default"):
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        if session_id in self.sessions:
            past_cache = self.sessions[session_id]
            past_len = past_cache.get_seq_length()
        else:
            past_cache = DynamicCache()
            past_len = 0

        total_len = past_len + seq_len
        position_ids = torch.arange(past_len, total_len, device=device).unsqueeze(0)  # [1, seq_len]
        cache_position = torch.arange(past_len, total_len, device=device)             # [seq_len]

        # Build causal attention mask
        attention_mask = torch.ones((batch_size, total_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=hidden_states,
            past_key_values_length=past_len,
            sliding_window=getattr(self.base.config, "sliding_window", None),
        )

        position_embeddings = self.base.model.rotary_emb(hidden_states, position_ids)
        current_hidden = hidden_states
        print(hidden_states.shape)

        # Forward through each backend layer
        for layer in self.base.model.layers:
            layer_device = layer.input_layernorm.weight.device
            # Move tensors to layer's device
            layer_hidden = current_hidden.to(layer_device)
            layer_position_ids = position_ids.to(layer_device)
            layer_attention_mask = attention_mask.to(layer_device)
            layer_position_embeddings = (position_embeddings[0].to(layer_device),position_embeddings[1].to(layer_device))
            layer_cache_position = cache_position.to(layer_device)

            layer_output = layer(
                hidden_states=layer_hidden,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                position_embeddings=layer_position_embeddings,
                past_key_values=past_cache,     
                use_cache=True,
                cache_position=layer_cache_position,
            )
            current_hidden = layer_output  

        final_hidden = self.base.model.norm(current_hidden.to(self.base.model.norm.weight.device))

        self.sessions[session_id] = past_cache
        return final_hidden

    def clear_session(self, session_id: str):
        """Clear KV cache for a session."""
        self.sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        return session_id in self.sessions

def add_noise(embeddings, noise_factor=0.5):
    norm_x = torch.norm(embeddings)
    norm_y = noise_factor * norm_x
    noise = torch.randn_like(embeddings)
    noise_unit = noise / torch.norm(noise)
    noise_scaled = noise_unit * norm_y
    return embeddings + noise_scaled

class SplitLLMGenerator:
    def __init__(self, model_name, client_head, model_backend, client_tail):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.client_head = client_head
        self.model_backend = model_backend
        self.client_tail = client_tail
        self.SYSTEM_PROMPT = "You are a helpful AI assistant."

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_k: Optional[int] = 50,
        top_p: float = 0.9,
        l2_norm: float = 0.0,
        session_id: Optional[str] = None,
    ):
        if session_id is None:
            session_id = str(uuid.uuid4())

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to('cuda:0')
        total_len = input_ids.shape[1]

        # Timing accumulators
        client_time = 0.0
        server_time = 0.0

        # === Prefill Step ===
        torch.cuda.synchronize()  # 确保 GPU 操作完成
        t0 = time.time()
        head_out = self.client_head.forward(input_ids, session_id=session_id)
        torch.cuda.synchronize()
        client_time += time.time() - t0

        if l2_norm > 0:
            head_out = add_noise(head_out, noise_factor=l2_norm)

        torch.cuda.synchronize()
        t0 = time.time()
        backend_out = self.model_backend.forward(head_out, session_id=session_id)
        torch.cuda.synchronize()
        server_time += time.time() - t0

        logits = self.client_tail.forward(backend_out.to('cuda:0'))
        next_token_logits = logits[:, -1, :] / temperature

        # Sampling (not timed, usually negligible)
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

        for step in range(1, max_new_tokens):
            new_token_id = next_token

            torch.cuda.synchronize()
            t0 = time.time()
            head_out = self.client_head.forward(new_token_id, session_id=session_id)
            torch.cuda.synchronize()
            client_time += time.time() - t0

            if l2_norm > 0:
                head_out = add_noise(head_out, noise_factor=l2_norm)

            torch.cuda.synchronize()
            t0 = time.time()
            backend_out = self.model_backend.forward(head_out, session_id=session_id)
            torch.cuda.synchronize()
            server_time += time.time() - t0

            logits = self.client_tail.forward(backend_out.to('cuda:0'))
            next_token_logits = logits[:, -1, :] / temperature

            # Sampling
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

        total_steps = total_len - input_ids.shape[1]  # number of generated tokens
        total_time = client_time + server_time
        tokens_per_second = total_steps / total_time
        client_token_per_second = total_steps / client_time if client_time > 0 else 0.0
        server_token_per_second = total_steps / server_time if server_time > 0 else 0.0
        print(f"\n[Timing] Generated {total_steps} tokens")
        print(f"[Timing] ClientHead total: {client_time:.3f}s | {client_time/total_steps:.3f}s/token | {client_token_per_second:.2f} tokens/s")
        print(f"[Timing] ModelBackend total: {server_time:.3f}s | {server_time/total_steps:.3f}s/token | {server_token_per_second:.2f} tokens/s")
        print(f"[Timing] Total inference: {total_time:.3f}s | {total_time/total_steps:.3f}s/token | {tokens_per_second:.2f} tokens/s")
        return response
    
def warmup_split_llm(generator, warmup_prompt="Warmup test", max_new_tokens=100, n_warmup_runs=2):
    print("\n=== 开始模型预热 ===")
    start_warmup = time.time()
    
    with torch.inference_mode():  # 禁用梯度，加速预热
        for i in range(n_warmup_runs):
            print(f"预热轮数 {i+1}/{n_warmup_runs}")
            _ = generator.generate(
                prompt=warmup_prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,  # 固定采样，减少预热耗时
                top_p=1.0,
                l2_norm=0  # 预热时关闭噪声
            )
    warmup_time = time.time() - start_warmup
    print(f"=== 预热完成！总耗时: {warmup_time:.2f}秒 ===\n")

if __name__ == "__main__":
    model_name = "/home/yueshuaibing/models/Qwen-2.5-7B"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    client_attention_layers = 3
    total_layers = config.num_hidden_layers

    clear_gpu()
    print(f"Loading model...")

    client_head = ClientHead(model_name, client_attention_layers, total_layers)
    client_tail = ClientTail(model_name, client_attention_layers, total_layers)
    model_backend = ModelBackend(model_name, client_attention_layers, total_layers)

    generator = SplitLLMGenerator(model_name, client_head, model_backend, client_tail)
    warmup_split_llm(generator)

    # Test
    prompt = "Hello! How are you?"
    response = generator.generate(prompt, max_new_tokens=40, temperature=1)
    print("\nResponse:\n", response)