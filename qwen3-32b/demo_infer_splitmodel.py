#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU_ISI
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU_ISI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
from transformers import AutoTokenizer
import transformers
import torch
from torch import nn

from transformers import (
    AutoTokenizer,
    Qwen3Config,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from modelsplit import QwenModel_Client, QwenModel_Server
import os
import time

from utils import load_pretrain, load_pretrain_split, load_client_pretrain, load_lm_head_pretrain, load_server_pretrain

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Load model configuration and tokenizer
model_name = "/opt/models/Qwen3-32B"
model_layers_name = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"
configuration = Qwen3Config.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_layers=configuration.num_hidden_layers  #64
client_layers=2

model_client = QwenModel_Client(configuration, client_layers)
model_server = QwenModel_Server(configuration, client_layers)
lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
#model_client, model_server, lm_head = load_pretrain_split(model_client, model_server, lm_head, model_name)
print("Loading split pre-trained weights...")
model_client = load_client_pretrain(model_client, model_layers_name,  total_layers, client_layers)
model_server = load_server_pretrain(model_server, model_layers_name, total_layers, client_layers)
lm_head = load_lm_head_pretrain(lm_head, model_layers_name)

model_client = model_client.half().cuda(0)
#model_server = model_server.half().cuda(1)
for name, param in model_server.named_parameters():
    if any(f'layers.{i}.' in name for i in range(30)): 
        param.data = param.data.half().to('cuda:0')
    else:  
        param.data = param.data.half().to('cuda:1')
lm_head = lm_head.half().cuda(1)

input_sentence = "Who is Crayon Shinchan?\n"
model_client.eval()
model_server.eval()
inputs = tokenizer(input_sentence, return_tensors='pt').to('cuda')

print("Split inference token by token:")
with torch.no_grad():
    start_time=time.time()
    for i in range(128):
        hidden_states, causal_mask, position_ids = model_client(**inputs)
        outputs = model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
        logits = lm_head(outputs[0])
        last_token_logits = logits[:, -1, :]
        predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
        predicted_token = tokenizer.decode(predicted_token_id)
        # import pdb; pdb.set_trace()
        input_sentence = input_sentence + predicted_token
        print(input_sentence)
        inputs = tokenizer(input_sentence, return_tensors='pt').to('cuda')
    print(f"Time cost: {time.time()-start_time}")
