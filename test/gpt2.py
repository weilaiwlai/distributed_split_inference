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

"""
export MASTER_ADDR="129.82.45.26"  # Master Node IP Address for teal
module purge
module load python/anaconda
"""

class GPT2LMHEAD(GPT2LMHeadModel):
    def __init__(self, config, num_layers):
        super().__init__(config)
        self.config.output_hidden_states = True
        self.transformer.h = self.transformer.h[:num_layers]
        self.transformer.ln_f = torch.nn.Identity()
        self.lm_head = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.hidden_states[-1]


class GPT2LMTAIL(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        full_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='model_store')
        self.lm_head = full_model.lm_head
        del full_model

    def forward(self, hidden_states):
        logits = self.lm_head(hidden_states)
        return logits


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
        os.environ['NCCL_SOCKET_IFNAME']=str(args.ifname)

        

        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])

        print(f'distributed setup: rank: {self.rank}, local_rank: {self.local_rank}, world size: {self.world_size}, master_address: {args.master_address}, master_port: {args.master_port} ')

        if args.device=='cuda' and torch.cuda.is_available():
            print('starting nccl backend (GPU)')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend='nccl',  init_method=f'env://')
        else:
            print('starting gloo backend (CPU)')
            dist.init_process_group(backend='gloo',  init_method=f'env://')
            self.device = 'cpu'
        
        print(f'distributed environment created successfully with device: {self.device}')


        self.data_owner_ranks = [i for i in range(self.data_owner_total_procs)]
        self.all_ranks = [i for i in range(self.data_owner_total_procs + self.model_owner_total_procs)]

        self.group_all = dist.new_group(ranks=self.all_ranks)
        self.group_do = dist.new_group(ranks=self.data_owner_ranks)

        dist.barrier(group=self.group_all)

        print(f"Process group initialized for rank {self.rank}, local rank: {self.local_rank}, world size: {self.world_size}.")
        print(f"Rank {dist.get_rank()} is part of group_do with ranks {self.data_owner_ranks}")
        
        self.tokenizer = self.initialize_tokenizer(model_name)
        self.model_head, self.model_tail = self.initialize_models(model_name, num_layers=1)

        dist.barrier(group=self.group_all)

    def initialize_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        return tokenizer

    def initialize_models(self, model_name, num_layers):

        model_head = GPT2LMHEAD.from_pretrained(model_name, num_layers=num_layers).to(self.device).eval()
        model_tail = GPT2LMTAIL(model_name).to(self.device).eval()
        return model_head, model_tail

    def model_communication(self, tokens):
        dist.send(torch.tensor([tokens['input_ids'].size(0)]).int().to(self.device), dst=1)
        dist.send(torch.tensor([tokens['input_ids'].size(1)]).int().to(self.device), dst=1)
        dist.barrier(group=self.group_all)

        hidden_states = self.model_head(**tokens)

        dist.send(hidden_states, dst=1)
        dist.barrier(group=self.group_all)

        hidden_states = torch.zeros_like(hidden_states).to(self.device)
        dist.recv(hidden_states, src=1)
        dist.barrier(group=self.group_all)

        output = self.model_tail(hidden_states)
        dist.barrier(group=self.group_all)

        return output

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=64, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self.model_communication(tokens)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            tokens_next = torch.multinomial(probs, num_samples=1)
            tokens['input_ids'] = torch.cat((tokens['input_ids'], tokens_next), dim=1)
            tokens['attention_mask'] = torch.cat(
                (tokens['attention_mask'], torch.ones((tokens['input_ids'].shape[0], 1), dtype=torch.long, device=self.device)),
                dim=1
            )
        return tokens

    def run(self, input_text, max_new_tokens=64, temperature=1.0, top_k=None):
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True,).to(self.device)
        if torch.cuda.is_available():
            tokens = {k: v.cuda() for k, v in tokens.items()}

        tokens_out = self.generate(tokens)
        responses = self.tokenizer.batch_decode(tokens_out["input_ids"], skip_special_tokens=True)
        return responses


'''
if __name__ == "__main__":
    executor = ModelExecutor(data_owner_total_procs=1, model_owner_total_procs=1, communication_rounds=1)
    input_texts = [
        "Hello, how are you today?", "What is the weather like?","I like traveling by train because"
    ]
    executor.run(input_texts)
'''