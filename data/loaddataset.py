import logging
from typing import List 
from datasets import load_dataset, Features, Value
import os

from data.prompts import get_prompt

log = logging.getLogger(__name__)  #创建日志记录器

"""
def get_random_prompted_examples(dataset, config) -> List[str]:    
        log.info(f"Loading dataset: {dataset}") 
        dataset_cache_dir = "../datasets/" + dataset,  #数据集地址
        cache_dir = dataset_cache_dir     
        examples = (
            load_dataset(
                path="parquet",  
                data_files={
                "test": "../datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",  # Parquet 文件路径
                },
                name=config.subset,
                split=config.split,
                cache_dir=cache_dir,  # 强制使用本地缓存目录
                download_mode="reuse_cache_if_exists"  
            )
            .shuffle(seed=config.random_seed)  #随机打乱
            .select(range(config.num_examples))  #从打乱后的数据集里选择指定数量
        )
        prompted_examples = [get_prompt(dataset, ex) for ex in examples]
        return prompted_examples
"""

def get_random_prompted_examples(dataset, config) -> list[str]:
        """Get random examples from the dataset and prompt them."""
        log.info(f"Loading dataset: {dataset}")
        dataset_cache_dir = os.path.abspath(os.path.join("..", "datasets", dataset))
        if not os.path.exists(dataset_cache_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_cache_dir}")
        examples = (
            load_dataset(
                path=dataset_cache_dir,
                name=config.subset,
                split=config.split,
                download_mode="reuse_cache_if_exists"
            )
            .shuffle(seed=config.random_seed)
            .select(range(config.num_examples))
        )
        prompted_examples = [get_prompt(dataset, ex) for ex in examples]
        return prompted_examples