import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import uuid
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import os
import gc
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def get_device_map(
    client_attention_layers=1,
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

class ModelBackend:
    def __init__(self, model_name, client_attention_layers, total_layers):
        self.device = torch.device("cuda:0")
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=False,
            tail=False,
            server_gpus=[0,1] 
        )
        
        # Load the full model first
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
            trust_remote_code=True
        )
        
        # Keep only the backend layers
        self.base = torch.nn.Module()
        self.base.model = torch.nn.Module()
        self.base.model.layers = full_model.model.layers[client_attention_layers:]
        self.base.model.norm = full_model.model.norm
        self.base.config = full_model.config
        
        # Disable lm_head (handled by ClientTail)
        self.base.lm_head = torch.nn.Identity()
        
        # Ensure rotary buffer is on a valid GPU (e.g., norm's device)
        norm_device = self.base.model.norm.weight.device
        if hasattr(full_model.model.rotary_emb, 'inv_freq'):
            self.base.model.rotary_emb = full_model.model.rotary_emb.to(norm_device)
        else:
            self.base.model.rotary_emb = full_model.model.rotary_emb

        # Session storage: session_id -> DynamicCache
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

        attention_mask = torch.ones((batch_size, total_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=hidden_states,
            past_key_values_length=past_len,
            sliding_window=getattr(self.base.config, "sliding_window", None),
        )

        cos, sin = self.base.model.rotary_emb(hidden_states, position_ids)
        current_hidden = hidden_states

        for layer in self.base.model.layers:
            layer_device = layer.input_layernorm.weight.device
            # Move tensors to layer's device
            layer_hidden = current_hidden.to(layer_device)
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
                past_key_values=past_cache,      # ← DynamicCache object
                use_cache=True,
                cache_position=layer_cache_position,
            )
            current_hidden = layer_output  

        final_hidden = self.base.model.norm(current_hidden.to(self.base.model.norm.weight.device))

        self.sessions[session_id] = past_cache
        return final_hidden

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        return session_id in self.sessions

class GenerationRequest(BaseModel):
    hidden_states: list
    session_id: Optional[str] = "default"

class GenerationResponse(BaseModel):
    hidden_states: list

def warmup_split_llm(model_backend: ModelBackend, warmup_prompt="Warmup test", n_warmup_runs=2):
    print("\n=== 开始模型预热 ===")
    
    hidden_size = model_backend.base.config.hidden_size
    batch_size = 1
    seq_len = 100  # 使用较短的序列进行预热
    
    for i in range(n_warmup_runs):
        print(f"正在进行第 {i+1}/{n_warmup_runs} 次预热运行...")
        
        # 创建随机的hidden states进行预热
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=model_backend.device)
        
        # 进行前向传播预热
        start_time = time.time()
        output = model_backend.forward(hidden_states, session_id=f"warmup_session_{i}")
        end_time = time.time()
        
        print(f"预热运行 {i+1} 完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"输出形状: {output.shape}")
    
    print("=== 模型预热完成 ===")

# Initialize model
model_name = "/opt/models/Qwen3-32B"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
client_attention_layers = 1
total_layers = config.num_hidden_layers


print(f"Loading model backend...")
model_backend = ModelBackend(model_name, client_attention_layers, total_layers)

warmup_split_llm(model_backend=model_backend)

app = FastAPI()

@app.post("/process_hidden_states")
async def process_hidden_states(request: GenerationRequest):
    try:
        # Convert the received list back to tensor, first to float32 then to bfloat16
        hidden_states = torch.tensor(request.hidden_states, dtype=torch.float32).bfloat16()
        session_id = request.session_id
        
        # Process through model backend
        output = model_backend.forward(hidden_states, session_id=session_id)
        
        # Convert tensor back to list for JSON response, first to float32 to avoid bfloat16 issues
        output_list = output.to(torch.float32).cpu().numpy().tolist()
        
        return GenerationResponse(hidden_states=output_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\n=== 开始启动FastAPI服务器 ===")
    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)