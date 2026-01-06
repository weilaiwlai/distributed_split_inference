from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable, List, Optional, Tuple, Union
#from accelerate import load_checkpoint_and_dispatch
import os
from torch.nn import functional as F
import gc
import time
import sys
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
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
    server_gpus=None  # 新增参数：指定服务器使用的 GPU 列表
):
    device_map = {}

    if server_gpus is None:
        server_gpus = [0, 1] 

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
        device_map[f"{attn}.k_proj.weight"] = device
        device_map[f"{attn}.v_proj.weight"] = device
        device_map[f"{attn}.o_proj.weight"] = device

        # === Qwen-specific: q_norm & k_norm ===
        device_map[f"{attn}.q_norm.weight"] = device
        device_map[f"{attn}.k_norm.weight"] = device

        # === MLP ===
        device_map[f"{mlp}.gate_proj.weight"] = device
        device_map[f"{mlp}.up_proj.weight"] = device
        device_map[f"{mlp}.down_proj.weight"] = device

        # === LayerNorms ===
        device_map[f"{prefix}.input_layernorm.weight"] = device
        device_map[f"{prefix}.post_attention_layernorm.weight"] = device

        # === Rotary Embedding buffer (safe to include) ===
        device_map[f"{attn}.rotary_emb.inv_freq"] = device

    return device_map

def list_meta_params(model):
    meta_params = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_params.append(name)
    return meta_params

class ClientHead:
    def __init__(self, ):
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
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        self.base.model.layers = self.base.model.layers[:client_attention_layers]

        client_device = self.base.model.embed_tokens.weight.device
        self.base.model.rotary_emb.inv_freq = self.base.model.rotary_emb.inv_freq.to(client_device)    
        #self.rotary_emb.to(self.running_device) 
    @torch.no_grad()
    def forward(self,  input_ids):
        #print("ClientHead - input_ids:", input_ids.shape)
        inputs_embeds = self.base.model.embed_tokens(input_ids)
        #position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)       
        cos, sin = self.base.model.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        batch_size, seq_length = hidden_states.shape[:2]
        attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length=0
        )
        for layer in self.base.model.layers:
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin)
            )
            hidden_states = layer_outputs
            #print(f"After layer : hidden_states.shape = {hidden_states.shape}")
        return hidden_states

class ClientTail:
    def __init__(self,):
        device_map = get_device_map(client_attention_layers=client_attention_layers, total_layers=total_layers, client=True, tail=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        self.lm_head=self.base.lm_head
        del self.base
    @torch.no_grad()
    def forward(self,  hidden_states):
        return self.lm_head(hidden_states)


class ModelBackend:
    def __init__(self, ):
        device_map = get_device_map(client_attention_layers=client_attention_layers, 
                                    total_layers=total_layers, client=False, tail=False)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        self.base.model.layers = self.base.model.layers[client_attention_layers:]
        self.base.lm_head=torch.nn.Identity()
        
        server_device = self.base.model.norm.weight.device
        self.base.model.rotary_emb.inv_freq = self.base.model.rotary_emb.inv_freq.to(server_device)
    @torch.no_grad()
    def forward(self, hidden_states):
        #position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        cos, sin = self.base.model.rotary_emb(hidden_states, position_ids)
                
        # Add attention mask
        batch_size, seq_length = hidden_states.shape[:2]
        attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length=0
            )
        for layer in self.base.model.layers:
            layer_device = layer.input_layernorm.weight.device
            hidden_states = hidden_states.to(layer_device)
            position_ids = position_ids.to(layer_device)
            attention_mask = attention_mask.to(layer_device)
            cos = cos.to(layer_device)
            sin = sin.to(layer_device)
            layer_outputs = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin) 
            )
            hidden_states = layer_outputs  # Get output of layer
            #print(f"After layer : hidden_states.shape = {hidden_states.shape}")
        hidden_states= self.base.model.norm(hidden_states)
        return hidden_states

def add_noise(embeddings,noise_factor=0.5,):
    norm_x = torch.norm(embeddings)   # Compute the norm of the matrix
    norm_y = noise_factor * norm_x  # Desired norm for the noise
    # Generate random noise matrix
    noise = torch.randn_like(embeddings)  # Same shape as M
    # Normalize the noise to have a unit norm
    noise_unit = noise / torch.norm(noise)
    # Scale the noise to have the desired norm
    noise_scaled = noise_unit * norm_y
    # Validate the noise norm
    #noise_norm = torch.norm(noise_scaled)
    #print(f"Desired noise norm: {norm_y}, Actual noise norm: {noise_norm}")
    #print(f'embedding; norm: {norm_x}')
    return embeddings+noise_scaled

class SplitLLMGenerator:
    def __init__(self, model_name, client_head, model_backend, client_tail):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.client_head = client_head
        self.model_backend = model_backend
        self.client_tail = client_tail
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.SYSTEM_PROMPT = "You are a helpful AI assistant. "  # Keep it simple

    def generate(self, prompt, max_new_tokens=128, temperature=0.6, top_k=50, top_p=0.9, l2_norm=0):
        torch.cuda.empty_cache()
        # Format the prompt for a single response
        #formatted_prompt = f"{self.SYSTEM_PROMPT}\nHuman: {prompt}\nAssistant: "
        formatted_prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to('cuda:0')
        current_input_ids = tokens.input_ids
        generated_tokens = current_input_ids

        for _ in range(max_new_tokens):
            # Forward pass through the split model
            head_output = self.client_head.forward(current_input_ids)
            #print("ClientHead - head_output.shape:", head_output.shape)
            if l2_norm > 0:
                head_output = add_noise(head_output, noise_factor=l2_norm)

            backend_output = self.model_backend.forward(head_output)
            #sys.exit()
            #print(backend_output.shape)
            backend_output = backend_output.to('cuda:0')
            logits = self.client_tail.forward(backend_output)
            #print(logits.shape)
            del head_output, backend_output

            # Apply temperature scaling
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                del top_k_logits, top_k_indices

            # Apply top-p filtering
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
                del sorted_logits, sorted_indices, cumulative_probs, sorted_indices_to_remove, indices_to_remove

            # Sample the next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            del probs, next_token_logits

            # Append the new token to the generated sequence
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            generated_tokens = current_input_ids

            # Decode the current sequence and check for "Human: "
            decoded_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            if "Human:" in decoded_text.split("Assistant:")[-1]:
                break

            # Check for termination tokens
            if next_token.item() in self.terminators:
                break

            del logits
            torch.cuda.empty_cache()

        # Decode and format the final response, removing "Human:"
        final_text = decoded_text.split("Assistant:")[-1].strip()
        final_text = final_text.split("Human:")[0].strip()  # Remove anything after "Human:"
        
        del tokens, current_input_ids, generated_tokens
        torch.cuda.empty_cache()
        return final_text

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
    #auth_token = os.getenv("HUGGINGFACE_TOKEN")
    model_name= "/opt/models/Qwen3-32B"

    device='cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    #print([n for n, _ in model.named_parameters()])  #

    config = AutoConfig.from_pretrained(model_name)
    client_attention_layers=2
    total_layers=config.num_hidden_layers

    clear_gpu()
    print(f"\n\n\nCurrent memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB")



    client_head=ClientHead()
    client_tail=ClientTail()
    model_backend=ModelBackend()

    print(client_head)
    print(client_tail)
    print(model_backend)


    print(client_head.base)
    # Example usage:
    meta_params = list_meta_params(client_head.base)
    print(f"Parameters on meta device: {meta_params}")

    client_head.base.eval()
    client_tail.lm_head.eval()
    model_backend.base.eval()
    print(f"\n\n\nCurrent memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB")

    generator = SplitLLMGenerator(model_name=model_name,  client_head=client_head, model_backend=model_backend, client_tail=client_tail)
    warmup_split_llm(generator)
    prompt = "Hey, how are you doing today?"
    max_new_tokens = 64

    start_time = time.time()
    response = generator.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    tokens_per_second = max_new_tokens / elapsed_time
    
    print('\n\nResponse:\n')
    print(response)
    print(f'\n\nInference Speed Test:')
    print(f'Prompt: "{prompt}"')
    print(f'Max new tokens: {max_new_tokens}')
    print(f'Elapsed time: {elapsed_time:.2f} seconds')
    print(f'Inference speed: {tokens_per_second:.2f} tokens/second')
    sys.exit()


    generator = SplitLLMGenerator(model_name=model_name,  client_head=client_head, model_backend=model_backend, client_tail=client_tail)
    response = generator.generate(prompt="Hey, how are you doing today?", max_new_tokens=10, temperature=1, )
    print('\n\nResponse:\n')
    print(response)

    response = generator.generate(prompt="I want to know the meaning of life. ", max_new_tokens=256, temperature=1, )
    print('\n\nResponse:\n')
    print(response)


    l2_norm=0.5
    generator = SplitLLMGenerator(model_name=model_name, client_head=client_head, model_backend=model_backend, client_tail=client_tail)
    response = generator.generate(prompt="Hey, how are you doing today?", max_new_tokens=256, temperature=1,l2_norm=l2_norm )
    print('\n\nResponse:\n')
    print(response)


    response = generator.generate(prompt="I want to know the meaning of life. ", max_new_tokens=256, temperature=1,l2_norm=l2_norm)
    print('\n\nResponse:\n')
    print(response)


    response = generator.generate(prompt="Write me a python class to get 1000000 prime numbers ", max_new_tokens=256, temperature=1,l2_norm=l2_norm)
    print('\n\nResponse:\n')
    print(response)