import torch
from client import ModelClient
from transformers import AutoTokenizer
from data import get_random_prompted_examples

class Config:
    def __init__(self):
        self.subset = "main" #2.0.0 None
        self.split = "test"
        self.num_examples = 100
        self.random_seed = 42

model_name = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"
client_layers=2

def client():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    client = ModelClient(
        model_name,
        client_layers,
        max_new_tokens=1024,
        addr="tcp://202.204.62.144:5558",
    )

    dataset = "gsm8k"  #cnn_dailymail mbpp openai_humaneval alpaca gsm8k
    config = Config()
    examples = get_random_prompted_examples(dataset, config)
    for i, example in enumerate(examples):
        print(f"\n=== Processing example {i+1}/{len(examples)} ===")
        print("Example:")
        print(example)
        messages = [{"role": "user", "content": example}]
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
        
        # 准备 prompt
        prompt_ids = prompt_ids.to(client.device)
        print("\n=== Prompt ===")
        print(prompt_ids.shape)

        # 发起生成
        result = client.generate(prompt_ids.unsqueeze(0))
        print("\n=== Generate Result ===")
        print(tokenizer.decode(result[0]))

    client.close()


if __name__ == "__main__":
    client()
