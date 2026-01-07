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
import zmq
import threading
from serial import MsgpackEncoder, MsgpackDecoder
from common import ReqHiddenStatesMessage, RespTokenIdMessage, ReqEndMessage, RespEndMessage
from utils import load_client_pretrain, load_lm_head_pretrain, load_server_pretrain
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Load model configuration and tokenizer
model_name = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"
client_layers=2
addr="tcp://0.0.0.0:5558"

class ModelServer:
    def __init__(self, 
                model_name:str, 
                client_layers:int,
                addr: str = "tcp://0.0.0.0:5558"):
        self.device = torch.device("cuda:0")

        self.configuration = Qwen3Config.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.total_layers=self.configuration.num_hidden_layers  #64
        self.model_split_layer=30

        self.model_server = QwenModel_Server(self.configuration, client_layers, self.model_split_layer)
        self.lm_head = nn.Linear(self.configuration.hidden_size, self.configuration.vocab_size, bias=False)
        print("Loading split pre-trained weights...")
        self.model_server = load_server_pretrain(self.model_server, model_name, self.total_layers, client_layers)
        self.lm_head = load_lm_head_pretrain(self.lm_head, model_name)

        for name, param in self.model_server.named_parameters():
            if any(f'layers.{i}.' in name for i in range(self.model_split_layer)): 
                param.data = param.data.half().to('cuda:0')
            else:  
                param.data = param.data.half().to('cuda:1')
        self.lm_head = self.lm_head.half().cuda(1)  

        self.model_server.eval()

        self.addr = addr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind(self.addr)

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(ReqHiddenStatesMessage | RespTokenIdMessage | ReqEndMessage)
        
        
        self.shutdown_event = threading.Event()
        
        self.server_thread = threading.Thread(target=self._server_loop, name="Server-Loop")
        self.server_thread.start()
    
    def handle_decode_request(self, msg)-> RespTokenIdMessage:
        """处理来自客户端的隐藏状态，执行前向传播并返回预测的token ID"""
        hidden_states = msg.hidden_states.to(self.device) if msg.hidden_states.device.type == 'cpu' else msg.hidden_states

        with torch.no_grad():
            outputs = self.model_server(
                hidden_states=hidden_states,
                use_cache=True,
            )
            logits = self.lm_head(outputs[0])
            last_token_logits = logits[:, -1, :]
            predicted_token_id = torch.argmax(last_token_logits, dim=-1)
            if predicted_token_id.numel() > 1:
                predicted_token_id = predicted_token_id.squeeze(0)  # 去除批次维度

        response = RespTokenIdMessage(
            request_id=msg.request_id,
            seq_id=msg.seq_id,
            predicted_token_id=predicted_token_id.cpu()
        )
        
        return response

    def _server_loop(self):
        print("[Server] main loop started.")
        while not self.shutdown_event.is_set():
            try:
                frames = self.sock.recv_multipart(copy=False, flags=zmq.NOBLOCK if self.shutdown_event.is_set() else 0)
            except zmq.Again:
                time.sleep(0.01)
                continue
            except zmq.ContextTerminated:
                break
            except Exception as e:
                print(f"[Server] recv failed: {e}")
                time.sleep(0.01)
                continue

            if not frames:
                continue
            
            client_id, *payload = frames
            try:
                msg = self.decoder.decode(payload)
                resp_msg = None
                
                if isinstance(msg, ReqHiddenStatesMessage):
                    resp_msg = self.handle_decode_request(msg)
                    print(f"[Server] received {type(msg).__name__} request_id={msg.request_id}, seq_id={msg.seq_id}")
                elif isinstance(msg, ReqEndMessage):
                    self.model_server.reset()
                    resp_msg = RespEndMessage(
                        request_id=msg.request_id,
                        seq_id=msg.seq_id,
                    )
                    print(f"[Server] received {type(msg).__name__} request_id={msg.request_id}, seq_id={msg.seq_id} and reset model server")
                else:
                    print(f"[Server] received unexpected message type: {type(msg)}")
                    continue

                if resp_msg is None:
                    continue
                
                payload = self.encoder.encode(resp_msg)
                frames = [client_id, *payload]
                try:
                    self.sock.send_multipart(frames, copy=False)
                    print(f"[Server] sent {type(resp_msg).__name__} request_id={resp_msg.request_id}, seq_id={resp_msg.seq_id}")
                except Exception as e:
                    print(f"[Server] send failed: {e}")
            except Exception as e:
                import traceback
                print(f"[Server] Error processing message: {e}")
                print(f"[Server] Full traceback: {traceback.format_exc()}")
                continue

    def close(self):
        self.shutdown_event.set()
        if hasattr(self, 'server_thread') and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        self.sock.close()
        self.ctx.term()

if __name__ == "__main__":
    server = ModelServer(model_name, client_layers, addr)
    try:
        # 保持服务器运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.close()