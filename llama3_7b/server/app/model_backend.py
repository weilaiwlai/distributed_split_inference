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

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import os
#from transformers.models.llama.modeling_llama import LlamaModel,LlamaForCausalLM,LlamaRotaryEmbedding,BaseModelOutputWithPast
from typing import Callable, List, Optional, Tuple, Union
#from accelerate import load_checkpoint_and_dispatch
import os
from torch.nn import functional as F
import gc
import sys
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

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


class ModelBackend:
    def __init__(self, ):
        device_map = get_device_map(client_attention_layers=client_attention_layers, 
                                    total_layers=total_layers, client=False, tail=False)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=auth_token,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='model_store', 
            cache_dir='model_store'
        )
        self.base.model.layers = self.base.model.layers[client_attention_layers:]
        self.base.lm_head=torch.nn.Identity()
        # Explicitly move rotary embedding components to GPU
        for layer in self.base.model.layers:
            layer.self_attn.rotary_emb.inv_freq = layer.self_attn.rotary_emb.inv_freq.cuda()
            # Clear any existing cache since it might be on wrong device
            if hasattr(layer.self_attn.rotary_emb, 'cos_cached'):
                layer.self_attn.rotary_emb.cos_cached=layer.self_attn.rotary_emb.cos_cached.cuda()
            if hasattr(layer.self_attn.rotary_emb, 'sin_cached'):
                layer.self_attn.rotary_emb.sin_cached=layer.self_attn.rotary_emb.sin_cached.cuda()
    @torch.no_grad()
    def forward(self,  hidden_states):
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        # Add attention mask
        batch_size, seq_length = hidden_states.shape[:2]
        attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length=0
            )
        for layer in self.base.model.layers:
            layer_outputs = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]  # Get output of layer
        hidden_states= self.base.model.norm(hidden_states)
        return hidden_states



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
    
    parser.add_argument('--model_name', type=str, default='gpt2', )

    parser.add_argument('--rank', type=int, default=1, )
    parser.add_argument('--local_rank', type=int, default=0, )
    parser.add_argument('--world_size', type=int, default=2, )

    parser.add_argument('--master_address', type=str, default="192.168.1.153", )
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

    #dist.barrier(group=group_all)
    generate(model_owner, device, rank, group_mo, group_all ,model_owner_initial_rank=1)
    

if __name__ == "__main__":
    #configs. 
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    model_name= "meta-llama/Meta-Llama-3-8B-Instruct"
    device='cuda:0'
    #tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    config = AutoConfig.from_pretrained(model_name, token=auth_token) 
    client_attention_layers=1
    total_layers=config.num_hidden_layers

    main()







