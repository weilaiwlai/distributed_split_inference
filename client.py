from transformers import AutoTokenizer
import transformers
import torch
from torch import nn

from transformers import (
    AutoTokenizer,
    Qwen3Config,
    LlamaConfig,
)
from qwen_modelsplit import QwenModel_Client
from llama_modelsplit import LlamaModel_Client
import queue
import os
import time
import zmq
from serial import MsgpackEncoder, MsgpackDecoder
import threading
import uuid
from typing import Any, Dict
from concurrent.futures import Future
from common import ReqHiddenStatesMessage,RespTokenIdMessage,ReqEndMessage,RespEndMessage
from utils import load_client_pretrain, load_lm_head_pretrain, load_server_pretrain
from metrics import Metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class ModelClient:
    def __init__(self, 
                model_name:str, 
                client_layers:int,
                max_new_tokens: int = 128,
                addr: str = "tcp://202.204.62.144:5558"):
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda:0")

        self.configuration = LlamaConfig.from_pretrained(model_name)
        self.total_layers=self.configuration.num_hidden_layers  #64

        self.model_client = LlamaModel_Client(self.configuration, client_layers, max_context_len=self.max_new_tokens)
        print("Loading split pre-trained weights...")
        self.model_client = load_client_pretrain(self.model_client, model_name, self.total_layers, client_layers)
        self.model_client = self.model_client.half().cuda(0)
        self.model_client.eval()

        self.addr = addr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.connect(self.addr)

        self.shutdown_event = threading.Event()

        # 发送队列存储 multipart frames（List[bytes]）
        self.req_queue: queue.Queue[tuple[ReqHiddenStatesMessage | RespTokenIdMessage, Future]] = queue.Queue()
        # pending 为 request_id -> Future 映射
        self.req_futures: Dict[str, Future] = {}

        self.sender_thread = threading.Thread(target=self._sender_loop, name="LLM-Sender")
        self.receiver_thread = threading.Thread(target=self._receiver_loop, name="LLM-Receiver")

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(ReqHiddenStatesMessage | RespTokenIdMessage | RespEndMessage)

        self.sender_thread.start()
        self.receiver_thread.start()

        self.metrics = Metrics()

    def _sender_loop(self):
        while not self.shutdown_event.is_set():
            try:
                msg, fut = self.req_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.req_futures[msg.request_id] = fut
                frames = self.encoder.encode(msg)
                self.sock.send_multipart(frames, copy=False)
            except Exception as e:
                print(f"[LLM Sender] send failed: {e}")

    def _receiver_loop(self):
        self.sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        while not self.shutdown_event.is_set():
            try:
                frames = self.sock.recv_multipart(copy=False)
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[LLM Receiver] recv failed: {e}")
                continue

            if not frames:
                continue
            
            # 解析消息
            msg = self.decoder.decode(frames)

            # 路由到相应 future
            fut = self.req_futures.pop(msg.request_id, None)
            if fut is not None:
                fut.set_result(msg)
            else:
                print(f"[LLM Receiver] unexpected message for request_id={msg.request_id}")

    def request_decode(
        self,
        seq_id: str,
        hidden_states: torch.Tensor,
    ):
        req_id = str(uuid.uuid4())
        fut = Future[RespTokenIdMessage]()

        req_msg = ReqHiddenStatesMessage(
            request_id=req_id,
            seq_id=seq_id,
            hidden_states=hidden_states.cpu(),
        )
        self.req_queue.put((req_msg, fut))
        return fut
    
    def request_end(
        self,
        seq_id: str,
    ):
        """发送结束请求，清理服务器端状态。"""
        req_id = str(uuid.uuid4())
        fut = Future[RespEndMessage]()

        req_msg = ReqEndMessage(
            request_id=req_id,
            seq_id=seq_id,
        )
        self.req_queue.put((req_msg, fut))
        return fut
    
    def prefill(self, input_ids, seq_id):
        """
        Prefill阶段：处理输入序列的全部token
        """
        self.metrics.start_generation()
        
        with torch.no_grad():
            hidden_states, causal_mask, position_ids = self.model_client(input_ids=input_ids)
            #print(f"prefill_hidden_states shape: {hidden_states.shape}")
            #print("position_ids",position_ids)
            fut_decode = self.request_decode(seq_id, hidden_states)
            msg_decode = fut_decode.result()
            predicted_token_id = msg_decode.predicted_token_id.to(self.device)
            self.metrics.record_first_token()
            
        return predicted_token_id

    def decode(self, input_ids, max_new_tokens, seq_id):   
        predicted_token_id = input_ids[:, -1]     
        with torch.no_grad():
            for i in range(max_new_tokens):
                hidden_states, causal_mask, position_ids = self.model_client(input_ids=predicted_token_id.unsqueeze(0))
                #print(f"decode_hidden_states shape: {hidden_states.shape}")
                #print("position_ids",position_ids)
                fut_decode = self.request_decode(seq_id, hidden_states)
                msg_decode = fut_decode.result()
                predicted_token_id = msg_decode.predicted_token_id.to(self.device)
                self.metrics.record_next_token()           
                if predicted_token_id.item() == self.configuration.eos_token_id:
                    print("Generated EOS token, stopping generation.")
                    break
                
                input_ids = torch.cat([input_ids, predicted_token_id.unsqueeze(0)], dim=-1)
        
        self.metrics.end_generation()
        return input_ids

    def generate(self, input_ids):
        #input_ids = inputs['input_ids'].to(self.device) 
        seq_id = str(uuid.uuid4())       
        first_token = self.prefill(input_ids, seq_id)
        
        input_ids = torch.cat([input_ids, first_token.unsqueeze(0)], dim=-1)     
        output_ids = self.decode(input_ids, self.max_new_tokens - 1, seq_id)

        fut_end = self.request_end(seq_id)
        self.model_client.reset()
        msg_end = fut_end.result()
        
        self.metrics.print_metrics()
        self.metrics.save_metrics_to_file()
        return output_ids

    
    def close(self):
        self.shutdown_event.set()
        self.sender_thread.join()
        self.receiver_thread.join()
        self.ctx.destroy()

if __name__ == "__main__":
    #model_name = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"
    model_name = "/home/yueshuaibing/models/Llama-3.1-70B/layers_safetensors"
    client_layers=3
    input_sentence = "Who is Crayon Shinchan?\n"
    model=ModelClient(model_name, client_layers, max_new_tokens=256, addr="tcp://202.204.62.144:5558")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_sentence, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    output=model.generate(input_ids)
    print(tokenizer.decode(output[0]))
    model.close()