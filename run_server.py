from server import ModelServer
import time

model_name = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"
client_layers=2
addr="tcp://0.0.0.0:5558"

if __name__ == "__main__":
    server = ModelServer(model_name, client_layers, addr)
    try:
        # 保持服务器运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.close()