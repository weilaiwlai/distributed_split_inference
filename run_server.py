from server import ModelServer
import time

model_name = "/home/yueshuaibing/models/Llama-3.1-70B/layers_safetensors"
is_large_model=True
client_layers=3
addr="tcp://0.0.0.0:5558"

if __name__ == "__main__":
    server = ModelServer(model_name, client_layers, addr, is_large_model)
    try:
        # 保持服务器运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.close()