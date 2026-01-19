from transformers import AutoTokenizer
import transformers
import torch
from torch import nn

from transformers import (
    AutoTokenizer,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from modelsplit import LlamaModel_Client, LlamaModel_Server
import os
import time

from utils import load_pretrain, load_pretrain_split, load_client_pretrain, load_lm_head_pretrain, load_server_pretrain

# Load model configuration and tokenizer
model_name = "/opt/models/Meta-Llama-3.1-70B-Instruct"
model_layers_name = "/home/yueshuaibing/models/Llama-3.1-70B/layers_safetensors"
configuration = LlamaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_layers=configuration.num_hidden_layers 
client_layers=3

model_client = LlamaModel_Client(configuration, client_layers)
model_server = LlamaModel_Server(configuration, client_layers, model_split_layer=20)
lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
#model_client, model_server, lm_head = load_pretrain_split(model_client, model_server, lm_head, model_name)
print("Loading split pre-trained weights...")
model_client = load_client_pretrain(model_client, model_layers_name,  total_layers, client_layers)
model_server = load_server_pretrain(model_server, model_layers_name, total_layers, client_layers)
lm_head = load_lm_head_pretrain(lm_head, model_layers_name)

model_client = model_client.half().cuda(0)
#model_server = model_server.half().cuda(1)
lm_head = lm_head.half().cuda(3)

input_sentence = "Who is Crayon Shinchan?\n"
model_client.eval()
model_server.eval()
inputs = tokenizer(input_sentence, return_tensors='pt').to('cuda')
input_ids = inputs['input_ids'].to('cuda')
output_ids = input_ids.clone()

print("Split inference token by token:")
with torch.no_grad():
    start_time=time.time()
    for i in range(1024):
        hidden_states, causal_mask, position_ids = model_client(input_ids=input_ids)
        outputs = model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
        logits = lm_head(outputs[0])
        last_token_logits = logits[:, -1, :]
        predicted_token_id = torch.argmax(last_token_logits, dim=-1)
        input_ids = predicted_token_id.unsqueeze(0).to('cuda:0')
        output_ids = torch.cat([output_ids, input_ids], dim=-1)
    print(f"Time cost: {time.time()-start_time}")
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs[0])
