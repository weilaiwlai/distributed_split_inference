import json
from pathlib import Path
from collections import defaultdict
import safetensors.torch
from tqdm import tqdm

MODEL_CONFIG = {
    'embed': 'model.embed_tokens',
    'layer_prefix': 'model.layers',
    'norm': 'model.norm',
    'lm_head': 'lm_head',
}

def preprocess_and_split_weights(model_path: str, config: dict):
    """
    根据提供的配置，读取一个大 safetensors 文件并将其分割成按层的小文件。
    """
    model_path = Path(model_path)
    output_dir = Path("/home/yueshuaibing/models/Llama-3.1-70B/layers_safetensors")
    output_dir.mkdir(exist_ok=True)

    print(f"Starting preprocessing for model: {model_path}")
    print(f"Output directory for split weights: {output_dir}")
    print(f"Using model configuration: {config}")

    # 查找权重文件
    source_path = model_path / "model.safetensors"
    if not source_path.exists():
        # 尝试查找分片模型的第一个文件（如果存在）
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            first_shard_filename = list(index_data['weight_map'].values())[0]
            print(
                f"Warning: Found sharded model index. This script is optimized for single files but will attempt to load all weights.")
            print("This may consume a large amount of CPU RAM.")
            # 对于分片模型，需要将所有分片加载并合并
            all_weights = {}
            shard_files = set(index_data['weight_map'].values())
            for shard_file in tqdm(shard_files, desc="Loading sharded weights"):
                shard_path = model_path / shard_file
                all_weights.update(safetensors.torch.load_file(shard_path, device="cpu"))
        else:
            raise FileNotFoundError(f"Cannot find model.safetensors or model.safetensors.index.json at {model_path}")
    else:
        # 单个 safetensors 文件
        print(f"Loading all weights from {source_path} to CPU memory for splitting...")
        all_weights = safetensors.torch.load_file(source_path, device="cpu")

    print(f"All weights loaded. Total tensors: {len(all_weights)}")

    # --- 按配置对权重键进行分组 ---
    grouped_weights = defaultdict(dict)
    unmatched_weights = {}

    layer_prefix = config['layer_prefix']

    for key, tensor in all_weights.items():
        if key.startswith(layer_prefix):
            # 提取层索引，例如 'model.layers.15.mlp.up_proj.weight' -> 'model.layers.15'
            # 这个逻辑对于个位数和多位数层号都有效
            layer_name = ".".join(key.split('.')[:3])
            grouped_weights[layer_name][key] = tensor
        elif key.startswith(config['embed']):
            grouped_weights[config['embed']][key] = tensor
        elif key.startswith(config['norm']):
            grouped_weights[config['norm']][key] = tensor
        elif key.startswith(config['lm_head']):
            grouped_weights[config['lm_head']][key] = tensor
        else:
            # 捕获所有未匹配的权重
            unmatched_weights[key] = tensor

    # --- 智能诊断和警告 ---
    if unmatched_weights:
        print("\n" + "=" * 50)
        print("!! WARNING: Some weights did not match the provided configuration.")
        print(f"These {len(unmatched_weights)} unmatched weights will be saved to 'misc.safetensors'.")
        print("Please check if these should belong to a specific layer:")
        for i, key in enumerate(list(unmatched_weights.keys())[:5]):  # 只显示前5个
            print(f"  - {key}")
        if len(unmatched_weights) > 5:
            print("  - ... and more.")
        print("You might need to adjust the MODEL_CONFIG at the top of the script.")
        print("=" * 50 + "\n")
        grouped_weights['misc'] = unmatched_weights

    print(f"Found {len(grouped_weights)} weight groups to save.")

    # --- 将每个组保存为单独的 safetensors 文件 ---
    for group_name, tensors_dict in tqdm(grouped_weights.items(), desc="Saving split weights"):
        safe_filename = group_name + '.safetensors'
        output_path = output_dir / safe_filename
        safetensors.torch.save_file(tensors_dict, output_path)

    print("\nPreprocessing complete!")
    print(f"Split weight files are saved in: {output_dir}")


if __name__ == '__main__':
    # ------------------ 使用前请修改此路径 ------------------
    MODEL_PATH = "/opt/models/Meta-Llama-3.1-70B-Instruct"# <--- 修改这里
    # ---------------------------------------------------------

    if not Path(MODEL_PATH).exists() or not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Model path does not exist or is not a directory: {MODEL_PATH}")

    preprocess_and_split_weights(MODEL_PATH, MODEL_CONFIG)