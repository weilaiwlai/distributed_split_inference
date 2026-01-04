import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)
import datetime
import os
import gc
import sys
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
        device_map = get_device_map(
            client_attention_layers=client_attention_layers,
            total_layers=total_layers,
            client=False,
            tail=False,
            server_gpus=[0,1] 
        )
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='/opt/models/Qwen3-32B',
            cache_dir='/opt/models/Qwen3-32B',
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

        cos, sin = self.base.model.rotary_emb(hidden_states, position_ids)
        current_hidden = hidden_states

        # Forward through each backend layer
        for layer in self.base.model.layers:
            layer_device = layer.input_layernorm.weight.device
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


def generate(model_owner,device,  rank, group_mo, group_all, model_owner_initial_rank=1):
    curr_batch_size=torch.zeros(1).int().to(device)
    curr_context_size=torch.zeros(1).int().to(device)
    print("generate...")
    while True:
        #at the end of training and test loops, the batch size will sometimes be smaller (since there arent many samples left. we need to communicate this. )
        if rank==model_owner_initial_rank:
            dist.recv(curr_batch_size, src=0,)

        dist.broadcast(curr_batch_size, src=model_owner_initial_rank, group=group_mo)
        #sending 0 from the data owner to simulate end of training.
        if curr_batch_size==0:
            break

        if rank==model_owner_initial_rank:
            dist.recv(curr_context_size, src=0,)
        dist.broadcast(curr_context_size, src=model_owner_initial_rank, group=group_mo)
        dist.barrier(group_all)

        data_owner_intermediate_output=nn.Parameter(torch.zeros(curr_batch_size[0],curr_context_size[0],
                                                                config.hidden_size,dtype=torch.bfloat16, ), requires_grad=False).to(device)
        if rank==model_owner_initial_rank:
            dist.recv(data_owner_intermediate_output, src=0,)
        dist.broadcast(data_owner_intermediate_output, src=model_owner_initial_rank,group=group_mo)
        dist.barrier(group_all)
        
        #data_owner_intermediate_output.retain_grad()
        model_owner_output = model_owner.forward(data_owner_intermediate_output,)
        if rank==model_owner_initial_rank:
            dist.send(model_owner_output, dst=0)

        dist.barrier(group_all)
        dist.barrier(group_all)


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_owner_total_procs', type=int, default=1 )
    parser.add_argument('--model_owner_total_procs', type=int, default=1 )
    parser.add_argument('--communication_rounds', type=int, default=1, )
    
    parser.add_argument('--model_name', type=str, default='/opt/models/Qwen3-32B', )

    parser.add_argument('--rank', type=int, default=1, )
    parser.add_argument('--local_rank', type=int, default=0, )
    parser.add_argument('--world_size', type=int, default=2, )

    parser.add_argument('--master_address', type=str, default="0,0,0,0", )
    parser.add_argument('--master_port', type=str, default="29500", )
    parser.add_argument('--device', type=str, default="cuda", )
    parser.add_argument('--ifname', type=str, default="ens5", )
    
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["RANK"] = str(args.rank)
    #s.environ["NCCL_SOCKET_IFNAME"] = str(args.ifname)

    #os.environ["NCCL_DEBUG"] = "INFO"

    #os.environ['NCCL_IB_DISABLE']='1'
    #os.environ['NCCL_SOCKET_NTHREADS'] = '1' 

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    
    data_owner_total_procs=args.data_owner_total_procs
    model_owner_initial_rank=data_owner_total_procs

    print(f'distributed setup: rank: {rank},   local_rank: {local_rank},  world size: {world_size}, master_address: {args.master_address},   master_port: {args.master_port} , ifname: {args.ifname}')

    if args.device=='cuda' and torch.cuda.is_available():
        print('starting nccl backend (GPU)')
        #os.environ['NCCL_SOCKET_IFNAME']=str(args.ifname)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',  init_method=f'env://',)
        device = torch.device(f"cuda:{local_rank}")
    else:
        print('starting gloo backend (CPU)')
        #os.environ['GLOO_SOCKET_IFNAME']=str(args.ifname)
        dist.init_process_group(backend='gloo',  init_method=f'env://',)
        device='cpu'

    print(f'distributed environment created successfully with device: {device}')
    model_owner_ranks=[i for i in range(model_owner_initial_rank, model_owner_initial_rank+args.model_owner_total_procs)]
    all_ranks=[i for i in range(args.data_owner_total_procs+args.model_owner_total_procs)]

    print(f'model owner ranks: {model_owner_ranks}')
    print(f'all_ranks ranks: {all_ranks}')
    group_all=dist.new_group(ranks=all_ranks)
    group_mo = dist.new_group(ranks=model_owner_ranks)
    dist.barrier(group=group_all)


    print(f"Process group initialized for rank {rank}, local rank: {local_rank} world size {world_size}.")

    model_owner = ModelBackend()
    model_owner.base.eval()

    dist.barrier(group=group_mo)
    dist.barrier(group=group_all)

    generate(model_owner, device, rank, group_mo, group_all ,model_owner_initial_rank=1)
    

if __name__ == "__main__":
    model_name= "/opt/models/Qwen3-32B"
    device='cuda:0'
    #tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    config = AutoConfig.from_pretrained(model_name) 
    client_attention_layers=1
    total_layers=config.num_hidden_layers

    main()