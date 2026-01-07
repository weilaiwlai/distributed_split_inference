# Distributed Split Inference

这是一个分布式模型推理系统，将大型语言模型（如Qwen3-32B）分割到客户端和服务器端进行推理，以优化资源使用和提高推理效率。

## 项目概述

本项目实现了模型分割推理（Model Split Inference），将大语言模型按层数分割成两部分：
- **客户端（Client）**: 运行模型的前几层，处理输入嵌入和初始层的计算
- **服务器端（Server）**: 运行模型的后续层和归一化层，完成剩余计算并输出结果

这种架构允许在资源受限的设备上运行大型语言模型，同时利用服务器的强大计算能力。

### 启动服务器
```bash
python run_server.py
```

### 运行客户端
```bash
python run_client.py
```

## 配置参数

- `model_name`: 模型路径（如`/opt/models/Qwen3-32B/layers_safetensors`）
- `client_layers`: 客户端运行的层数
- `max_new_tokens`: 最大生成token数
- `addr`: 通信地址（默认为`tcp://0.0.0.0:5558`）