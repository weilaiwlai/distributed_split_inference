
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import os
def get_device_map(client_attention_layers=4,total_layers=32, client=True, tail=False):
    device_map={}
    device_map = {
        "model.embed_tokens.weight": 0 if client and not tail else "disk",
        "model.norm.weight": 0 if not client and not tail else "disk",
        "lm_head.weight": 0 if client  and tail else "disk"
    }

    for i in range(client_attention_layers):
        device_map[f"model.layers.{i}.self_attn.q_proj.weight"] = 0 if client and not tail else "disk"
        device_map[f"model.layers.{i}.self_attn.k_proj.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.self_attn.v_proj.weight"] = 0 if client  and not tail else "disk"
        device_map[f"model.layers.{i}.self_attn.o_proj.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.mlp.gate_proj.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.mlp.up_proj.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.mlp.down_proj.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.input_layernorm.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.post_attention_layernorm.weight"] = 0 if client and not tail  else "disk"
        device_map[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = 0 if client and not tail  else "disk"
    for i in range(client_attention_layers,total_layers):
        device_map[f"model.layers.{i}.self_attn.q_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.self_attn.k_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.self_attn.v_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.self_attn.o_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.mlp.gate_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.mlp.up_proj.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.mlp.down_proj.weight"] = "disk" if client  else 0
        device_map[f"model.layers.{i}.input_layernorm.weight"] = "disk" if client   else 0
        device_map[f"model.layers.{i}.post_attention_layernorm.weight"] = "disk" if client  else 0
        device_map[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = "disk" if client   else 0
    return device_map
if __name__ == "__main__":
    #configs. 
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    model_name= "meta-llama/Meta-Llama-3-8B-Instruct"
    device='cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    config = AutoConfig.from_pretrained(model_name, token=auth_token) 
    client_attention_layers=1
    total_layers=config.num_hidden_layers


    device_map = get_device_map(client_attention_layers=client_attention_layers, total_layers=total_layers, client=True, tail=False)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        offload_folder='model_store/',
        cache_dir='model_store/'
    )



    print(base)