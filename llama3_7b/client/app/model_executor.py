import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from torch.nn import functional as F
import argparse
import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from transformers.models.llama.modeling_llama import LlamaModel,LlamaForCausalLM,LlamaRotaryEmbedding,BaseModelOutputWithPast
from typing import Callable, List, Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import os
from torch.nn import functional as F
import gc
import sys
import string
import asyncio
from collections import deque


auth_token = os.getenv("HUGGINGFACE_TOKEN")
model_name= "meta-llama/Meta-Llama-3-8B-Instruct"

device='cuda:0'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)


config = AutoConfig.from_pretrained(model_name, token=auth_token) 
client_attention_layers=1
total_layers=config.num_hidden_layers


def list_meta_params(model):
    meta_params = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_params.append(name)
    return meta_params


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




class ClientHead:
    def __init__(self, ):
        device_map = get_device_map(client_attention_layers=client_attention_layers, total_layers=total_layers, client=True, tail=False)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=auth_token,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='model_store',
            cache_dir='model_store'
        )
        self.base.model.layers = self.base.model.layers[:client_attention_layers]
        #self.base.lm_head=torch.nn.Identity()
        #self.base.model.norm = torch.nn.Identity()
        #del self.base.model.norm
        # Explicitly move rotary embedding components to GPU
        for layer in self.base.model.layers[:client_attention_layers]:
            layer.self_attn.rotary_emb.inv_freq = layer.self_attn.rotary_emb.inv_freq.to(device)
            # Clear any existing cache since it might be on wrong device
            if hasattr(layer.self_attn.rotary_emb, 'cos_cached'):
                layer.self_attn.rotary_emb.cos_cached=layer.self_attn.rotary_emb.cos_cached.to(device)
            if hasattr(layer.self_attn.rotary_emb, 'sin_cached'):
                layer.self_attn.rotary_emb.sin_cached=layer.self_attn.rotary_emb.sin_cached.to(device)

    @torch.no_grad()
    def forward(self,  input_ids):
        inputs_embeds = self.base.model.embed_tokens(input_ids)

        #position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
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
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class ClientTail:
    def __init__(self,):
        device_map = get_device_map(client_attention_layers=client_attention_layers, total_layers=total_layers, client=True, tail=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=auth_token,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            offload_folder='model_store',
            cache_dir='model_store'
        )
        self.lm_head=self.base.lm_head
        self.lm_head.eval()
        del self.base
    @torch.no_grad()
    def forward(self,  hidden_states):
        return self.lm_head(hidden_states)



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




class ModelExecutor:
    def __init__(self, args ):

        self.model_name=args.model_name

        data_owner_total_procs=1
        model_owner_total_procs=1
        communication_rounds=1

        self.data_owner_total_procs = data_owner_total_procs
        self.model_owner_total_procs = model_owner_total_procs
        self.communication_rounds = communication_rounds


        os.environ["MASTER_ADDR"] = args.master_address
        os.environ["MASTER_PORT"] = args.master_port
        
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        os.environ["RANK"] = str(args.rank)

        #don't really need this unless you are desperate for bug clues
        os.environ["NCCL_DEBUG"] = "INFO"

        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])

        print(f'distributed setup: rank: {self.rank}, local_rank: {self.local_rank}, world size: {self.world_size}, master_address: {os.environ["MASTER_ADDR"]}, master_port: {os.environ["MASTER_PORT"]} ')

        if args.device=='cuda' and torch.cuda.is_available():
            print('starting nccl backend (GPU)')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend='nccl',  init_method=f'env://', )
        else:
            print('starting gloo backend (CPU)')
            dist.init_process_group(backend='gloo',  init_method=f'env://',)
            self.device = 'cpu'

        
        print(f'distributed environment created successfully with device: {self.device}')


        self.data_owner_ranks = [0]
        self.all_ranks = [i for i in range(self.data_owner_total_procs + self.model_owner_total_procs)]

        self.group_all = dist.new_group(ranks=self.all_ranks)
        self.group_do = dist.new_group(ranks=self.data_owner_ranks)

        dist.barrier(group=self.group_all)

        print(f"Process group initialized for rank {self.rank}, local rank: {self.local_rank}, world size: {self.world_size}.")
        print(f"Rank {dist.get_rank()} is part of group_do with ranks {self.data_owner_ranks}")
        

        auth_token = os.getenv("HUGGINGFACE_TOKEN")
        model_name= "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)

        self.model_head, self.model_tail = self.initialize_models(self.model_name, num_layers=1)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.eos_token, 
        ]
        self.SYSTEM_PROMPT = "You are a helpful AI assistant. Please end your responses with <|eot_id|>. The following is a conversation between a user and an assistant."  # Keep it simple

        dist.barrier(group=self.group_all)

    def initialize_models(self, model_name, num_layers):
        client_head=ClientHead()
        client_tail=ClientTail()

        client_head.base.eval()
        #client_tail.eval()
        return client_head, client_tail

    def model_communication(self, tokens, l2_norm=0):
        dist.send(torch.tensor([tokens.size(0)]).int().to(self.device), dst=1)
        dist.send(torch.tensor([tokens.size(1)]).int().to(self.device), dst=1)
        dist.barrier(group=self.group_all)

        hidden_states = self.model_head.forward(tokens)
        if l2_norm>0:
            hidden_states=add_noise(hidden_states,noise_factor=l2_norm,)
        hidden_states=hidden_states.to(self.device)

        dist.send(hidden_states, dst=1)
        dist.barrier(group=self.group_all)

        hidden_states = torch.zeros_like(hidden_states).to(self.device)
        dist.recv(hidden_states, src=1)
        dist.barrier(group=self.group_all)

        output = self.model_tail.forward(hidden_states)
        dist.barrier(group=self.group_all)
        return output

    @torch.no_grad()
    def run(self, input_text, max_new_tokens=128, temperature=0.6, top_k=50, top_p=0.9, l2_norm=0 ):
        torch.cuda.empty_cache()

        # Format the prompt for a single response
        formatted_prompt = f"{self.SYSTEM_PROMPT}\nHuman: {input_text}  <|eot_id|>\nAssistant: "
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to('cuda:0')
        current_input_ids = tokens.input_ids.to(device)
        generated_tokens = current_input_ids
        for _ in range(max_new_tokens):
            # Forward pass through the split model
            logits=self.model_communication(current_input_ids, l2_norm)

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
            #torch.cuda.empty_cache()

        
        print(decoded_text)

        # Decode and format the final response, removing "Human:"
        final_text = decoded_text.split("Assistant:")[-1].strip()
        final_text = final_text.split("Human:")[0].strip()  # Remove anything after "Human:"
        
        del tokens, current_input_ids, generated_tokens
        torch.cuda.empty_cache()
        return final_text

    @torch.no_grad()
    async def stream_run(
        self,
        input_text: str,
        history: str,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.9,
        l2_norm: float = 0
    ):
        """
        Async generator that streams tokens as soon as they are produced,
        incorporating all logic from the `run` method and handling termination.
        """
        import string

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Format the prompt for a single response
        formatted_prompt = f"{self.SYSTEM_PROMPT}\nHistory: {history}\nAssistant: "
        tokens = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        current_input_ids = tokens.input_ids.to(self.device)
        generated_tokens = current_input_ids

        # Initialize a string to keep track of the generated text
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        for _ in range(max_new_tokens):
            # Forward pass through the model
            #logits = self.model(current_input_ids).logits
            logits=self.model_communication(current_input_ids, l2_norm)

            # Apply temperature scaling
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                del top_k_logits, top_k_indices

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
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

            # Decode the new token
            just_added = self.tokenizer.decode(next_token[0], skip_special_tokens=True)

            # Append to generated_text
            generated_text += just_added


            if next_token.item() in self.terminators:
                break

            # Yield the token
            yield just_added


            # Yield control to allow other coroutines to run
            await asyncio.sleep(0)

        # Final cleanup
        torch.cuda.empty_cache()