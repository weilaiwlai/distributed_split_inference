import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen2_7b_model_split_kvcache import ClientHead  

model_name = "/home/yueshuaibing/models/Qwen-2.5-7B"
client_head = ClientHead(model_name, client_attention_layers=1, total_layers=32)

# Warmup
dummy_input = torch.randint(0, 10000, (1, 1), device="cuda:0")
_ = client_head.forward(dummy_input, "warmup")

# Pure timing of ClientHead only
times = []
for i in range(10):
    x = torch.randint(0, 10000, (1, 2000), device="cuda:0")
    torch.cuda.synchronize()
    t0 = time.time()
    out = client_head.forward(x, f"run_{i}")
    torch.cuda.synchronize()
    times.append(time.time() - t0)

avg_ms = sum(times) / len(times) * 1000
print(f"✅ Pure ClientHead avg latency: {avg_ms:.2f} ms")

# Dummy input
input_ids = torch.randint(0, 10000, (1, 1), device="cuda:0")

# Warmup
_ = client_head.forward(input_ids, "test")

# Measure
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1):
    out = client_head.forward(input_ids, "test")
torch.cuda.synchronize()
print("Avg time:", (time.time() - t0) / 1)