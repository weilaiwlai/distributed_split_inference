import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 启用两张显卡
import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, DynamicCache
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from safetensors.torch import load_file
import gc

# ===================== 全局配置（适配你的环境） =====================
MODEL_PATH = "/opt/models/Qwen3-32B"
LAYERS_SAFETENSORS_PATH = "/home/yueshuaibing/models/Qwen3-32B/layers_safetensors"  # 权重文件目录
DTYPE = torch.float16

# 显卡部署配置
EMBED_FIRST_LAYER_DEVICE = "cuda:0"  # 第一个类的显卡
TRANSFORMER_DEVICE1 = "cuda:0"       # 第二个类的第一张卡
TRANSFORMER_DEVICE2 = "cuda:1"       # 第二个类的第二张卡
NORM_LMHEAD_DEVICE = "cuda:0"        # 第三个类的显卡

# 层名称映射（匹配你的67个文件命名）
LAYER_NAME_MAP = {
    "embed": "model.embed_tokens",
    "transformer": "model.layers",
    "norm": "model.norm",
    "lm_head": "lm_head"
}

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ===================== 工具函数 =====================
def get_position_ids(input_ids, padding_idx=0):
    """生成有效的position_ids（适配Qwen3的RoPE）"""
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices - 1

def move_tuple_to_device(tuple_data, device):
    """将元组中的所有张量移动到指定设备（适配Qwen3的RoPE返回值）"""
    if isinstance(tuple_data, tuple):
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tuple_data)
    elif isinstance(tuple_data, torch.Tensor):
        return tuple_data.to(device)
    else:
        return tuple_data

def prepare_attention_mask_for_generation(attention_mask, input_shape, device):
    batch_size, seq_length = input_shape
    
    if attention_mask is not None:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
            attention_mask = attention_mask.unsqueeze(1)  # 变为4D: [batch, 1, seq_len, original_len]
        
        if attention_mask.size(-1) < seq_length:
            pad_length = seq_length - attention_mask.size(-1)
            pad = torch.ones(
                *attention_mask.shape[:-1], pad_length,
                device=device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, pad], dim=-1)

        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=device)).view(
            1, 1, seq_length, seq_length
        )
        
        attention_mask = attention_mask * causal_mask
        attention_mask = attention_mask.to(dtype=DTYPE)
    else:
        attention_mask = torch.tril(torch.ones((1, 1, seq_length, seq_length), device=device)).to(dtype=DTYPE)
    
    return attention_mask

# ===================== 第一个类：Embedding + 第1层Transformer（索引0） =====================
class EmbeddingFirstLayer:
    def __init__(self, model_path, layers_safetensors_path, device=EMBED_FIRST_LAYER_DEVICE, dtype=DTYPE):
        self.model_path = model_path
        self.layers_safetensors_path = layers_safetensors_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = None  # 后续绑定
        
        # 1. 加载配置
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        
        # 2. 空权重初始化模型
        with init_empty_weights(include_buffers=False):
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        
        # 3. 初始化旋转编码（仅传config，匹配Qwen3源码）
        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)
        self.rotary_emb.to(self.device, dtype=self.dtype)
        self.model.model.rotary_emb = self.rotary_emb
        
        # 4. 加载并部署指定层（embed + transformer.0）
        self._load_and_deploy_layers()
        
        # 5. 设为评估模式
        self.model.eval()
        self.model.tie_weights()
        
        print(f"[EmbeddingFirstLayer] 初始化完成，部署在 {self.device}")
        print(f"  - 加载层：{LAYER_NAME_MAP['embed']} + {LAYER_NAME_MAP['transformer']}.0")

    def _load_and_deploy_layers(self):
        """加载指定层的独立safetensors文件并部署到显卡"""
        # ========== 1. 加载Embedding层 ==========
        embed_layer_name = LAYER_NAME_MAP["embed"]
        embed_file_path = Path(self.layers_safetensors_path) / f"{embed_layer_name}.safetensors"
        if not embed_file_path.exists():
            raise FileNotFoundError(f"Embedding层权重文件不存在：{embed_file_path}")
        embed_state_dict = load_file(embed_file_path, device="cpu")
        self._deploy_state_dict(embed_state_dict, self.device)
        
        # ========== 2. 加载Transformer第0层（第一层） ==========
        first_transformer_layer_name = f"{LAYER_NAME_MAP['transformer']}.0"
        first_transformer_file_path = Path(self.layers_safetensors_path) / f"{first_transformer_layer_name}.safetensors"
        if not first_transformer_file_path.exists():
            raise FileNotFoundError(f"Transformer第0层权重文件不存在：{first_transformer_file_path}")
        first_transformer_state_dict = load_file(first_transformer_file_path, device="cpu")
        self._deploy_state_dict(first_transformer_state_dict, self.device)
        
        # ========== 3. 部署所有buffer（包括rotary_emb的inv_freq） ==========
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.device,
                value=buffer, dtype=self.dtype
            )
        
        # 额外确保rotary_emb的inv_freq在正确设备上
        if hasattr(self.rotary_emb, 'inv_freq') and self.rotary_emb.inv_freq is not None:
            self.rotary_emb.inv_freq = self.rotary_emb.inv_freq.to(self.device)

    def _deploy_state_dict(self, state_dict, device):
        """将权重部署到指定显卡"""
        for param_name, tensor in state_dict.items():
            tensor = tensor.to(dtype=self.dtype)
            set_module_tensor_to_device(
                self.model, param_name, device,
                value=tensor, dtype=self.dtype
            )

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, past_key_values=None, cache_position=None):
        """前向推理：embedding + 第1层Transformer（修复增量生成掩码）"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer未绑定，请先调用pipeline初始化")
        
        # 1. 获取当前输入形状
        batch_size, seq_length = input_ids.shape
        
        # 2. 生成有效的position_ids
        position_ids = get_position_ids(input_ids, padding_idx=self.tokenizer.pad_token_id).to(self.device)
        
        # 3. Embedding层推理
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # 4. 准备正确维度的注意力掩码（核心修复）
        expanded_attn_mask = prepare_attention_mask_for_generation(
            attention_mask, (batch_size, seq_length), self.device
        )
        
        # 5. 计算位置编码（Qwen3返回tuple: (cos, sin)）
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 6. 第1层Transformer推理
        layer_outputs = self.model.model.layers[0](
            hidden_states=hidden_states,
            attention_mask=expanded_attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=True,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs
        
        return {
            "hidden_states": hidden_states,
            "expanded_attn_mask": expanded_attn_mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids
        }

# ===================== 第二个类：剩余Transformer层（索引1-63） =====================
class TransformerLayers:
    def __init__(self, model_path, layers_safetensors_path, 
                 device1=TRANSFORMER_DEVICE1, device2=TRANSFORMER_DEVICE2, dtype=DTYPE):
        self.model_path = model_path
        self.layers_safetensors_path = layers_safetensors_path
        self.device1 = device1  # 前N层的显卡
        self.device2 = device2  # 后M层的显卡
        self.dtype = dtype
        
        # 1. 加载配置
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        self.total_transformer_layers = self.config.num_hidden_layers  # 64层
        self.first_layer_idx = 1  # 从第1层开始（跳过第0层）
        self.last_layer_idx = self.total_transformer_layers - 1  # 63层
        self.split_idx = 32  # 拆分点：1-32层→device1，33-63层→device2
        print(f"[TransformerLayers] 拆分配置：")
        print(f"  - {self.first_layer_idx}-{self.split_idx}层 → {device1}")
        print(f"  - {self.split_idx+1}-{self.last_layer_idx}层 → {device2}")
        
        # 2. 空权重初始化模型
        with init_empty_weights(include_buffers=False):
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        
        # 3. 加载并部署剩余Transformer层
        self._load_and_deploy_layers()
        
        # 4. 设为评估模式
        self.model.eval()
        print(f"[TransformerLayers] 初始化完成，共加载 {self.last_layer_idx - self.first_layer_idx + 1} 层")

    def _load_and_deploy_layers(self):
        """加载剩余Transformer层的独立文件并部署到双卡"""
        # ========== 1. 部署1-32层到device1 ==========
        for layer_idx in range(self.first_layer_idx, self.split_idx + 1):
            layer_name = f"{LAYER_NAME_MAP['transformer']}.{layer_idx}"
            file_path = Path(self.layers_safetensors_path) / f"{layer_name}.safetensors"
            if not file_path.exists():
                raise FileNotFoundError(f"Transformer第{layer_idx}层权重文件不存在：{file_path}")
            state_dict = load_file(file_path, device="cpu")
            self._deploy_state_dict(state_dict, self.device1, layer_name)
        
        # ========== 2. 部署33-63层到device2 ==========
        for layer_idx in range(self.split_idx + 1, self.last_layer_idx + 1):
            layer_name = f"{LAYER_NAME_MAP['transformer']}.{layer_idx}"
            file_path = Path(self.layers_safetensors_path) / f"{layer_name}.safetensors"
            if not file_path.exists():
                raise FileNotFoundError(f"Transformer第{layer_idx}层权重文件不存在：{file_path}")
            state_dict = load_file(file_path, device="cpu")
            self._deploy_state_dict(state_dict, self.device2, layer_name)

    def _deploy_state_dict(self, state_dict, device, layer_name):
        """将指定层的权重部署到指定显卡"""
        for param_name, tensor in state_dict.items():
            if param_name.startswith(layer_name):
                tensor = tensor.to(dtype=self.dtype)
                set_module_tensor_to_device(
                    self.model, param_name, device,
                    value=tensor, dtype=self.dtype
                )

    @torch.inference_mode()
    def forward(self, hidden_states, expanded_attn_mask, position_embeddings, position_ids,
                attention_mask=None, past_key_values=None, cache_position=None):
        """前向推理剩余Transformer层（双卡）"""
        # ========== 1. 1-32层推理（device1） ==========
        hidden_states = hidden_states.to(self.device1)
        expanded_attn_mask = expanded_attn_mask.to(self.device1) if isinstance(expanded_attn_mask, torch.Tensor) else expanded_attn_mask
        position_embeddings = move_tuple_to_device(position_embeddings, self.device1)
        position_ids = position_ids.to(self.device1) if position_ids is not None else None
        
        for layer_idx in range(self.first_layer_idx, self.split_idx + 1):
            layer = self.model.model.layers[layer_idx]
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=expanded_attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs
        
        # ========== 2. 跨卡传输到device2 ==========
        hidden_states = hidden_states.to(self.device2, non_blocking=True)
        expanded_attn_mask = expanded_attn_mask.to(self.device2) if isinstance(expanded_attn_mask, torch.Tensor) else expanded_attn_mask
        position_embeddings = move_tuple_to_device(position_embeddings, self.device2)
        position_ids = position_ids.to(self.device2, non_blocking=True) if position_ids is not None else None
        
        # ========== 3. 33-63层推理（device2） ==========
        for layer_idx in range(self.split_idx + 1, self.last_layer_idx + 1):
            layer = self.model.model.layers[layer_idx]
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=expanded_attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs
        
        # ========== 4. 传输回device1（适配第三个类） ==========
        hidden_states = hidden_states.to(NORM_LMHEAD_DEVICE, non_blocking=True)
        
        return {"hidden_states": hidden_states}

# ===================== 第三个类：Norm + LM Head层 =====================
class NormLMHeadLayer:
    def __init__(self, model_path, layers_safetensors_path, device=NORM_LMHEAD_DEVICE, dtype=DTYPE):
        self.model_path = model_path
        self.layers_safetensors_path = layers_safetensors_path
        self.device = device
        self.dtype = dtype
        
        # 1. 加载配置
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        
        # 2. 空权重初始化模型
        with init_empty_weights(include_buffers=False):
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        
        # 3. 加载并部署指定层（norm + lm_head）
        self._load_and_deploy_layers()
        
        # 4. 设为评估模式
        self.model.eval()
        self.model.tie_weights()
        
        print(f"[NormLMHeadLayer] 初始化完成，部署在 {self.device}")
        print(f"  - 加载层：{LAYER_NAME_MAP['norm']} + {LAYER_NAME_MAP['lm_head']}")

    def _load_and_deploy_layers(self):
        """加载norm和lm_head层的独立文件并部署到显卡"""
        # ========== 1. 加载Norm层 ==========
        norm_layer_name = LAYER_NAME_MAP["norm"]
        norm_file_path = Path(self.layers_safetensors_path) / f"{norm_layer_name}.safetensors"
        if not norm_file_path.exists():
            raise FileNotFoundError(f"Norm层权重文件不存在：{norm_file_path}")
        norm_state_dict = load_file(norm_file_path, device="cpu")
        self._deploy_state_dict(norm_state_dict, self.device)
        
        # ========== 2. 加载LM Head层 ==========
        lm_head_layer_name = LAYER_NAME_MAP["lm_head"]
        lm_head_file_path = Path(self.layers_safetensors_path) / f"{lm_head_layer_name}.safetensors"
        if not lm_head_file_path.exists():
            raise FileNotFoundError(f"LM Head层权重文件不存在：{lm_head_file_path}")
        lm_head_state_dict = load_file(lm_head_file_path, device="cpu")
        self._deploy_state_dict(lm_head_state_dict, self.device)

    def _deploy_state_dict(self, state_dict, device):
        """将权重部署到指定显卡"""
        for param_name, tensor in state_dict.items():
            tensor = tensor.to(dtype=self.dtype)
            set_module_tensor_to_device(
                self.model, param_name, device,
                value=tensor, dtype=self.dtype
            )

    @torch.inference_mode()
    def forward(self, hidden_states):
        """前向推理：norm + lm_head"""
        # 1. Norm层推理
        hidden_states = self.model.model.norm(hidden_states)
        
        # 2. LM Head层推理（转float32保证精度）
        logits = self.model.lm_head(hidden_states).to(torch.float32)
        
        return {"logits": logits}

# ===================== 推理流水线 =====================
class Qwen3Pipeline:
    def __init__(self, model_path=MODEL_PATH, layers_safetensors_path=LAYERS_SAFETENSORS_PATH):
        # 1. 初始化Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            padding_side="left", use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 2. 初始化三个模块（按顺序加载）
        print("="*60)
        self.embed_first_layer = EmbeddingFirstLayer(model_path, layers_safetensors_path)
        # 给第一个类绑定tokenizer
        self.embed_first_layer.tokenizer = self.tokenizer
        print("="*60)
        self.transformer_layers = TransformerLayers(model_path, layers_safetensors_path)
        print("="*60)
        self.norm_lmhead_layer = NormLMHeadLayer(model_path, layers_safetensors_path)
        print("="*60)
        
        # 3. 初始化生成配置
        self.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

    def generate(self, input_texts, max_new_tokens=30):
        """生成文本的主函数（修复增量生成维度不匹配）"""
        # 1. 预处理输入
        input_texts = [
            self.tokenizer.apply_chat_template(
                [{"role":"user","content":text}],
                tokenize=False,
                add_generation_prompt=True
            ) for text in input_texts
        ]
        
        # 2. 分词
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = inputs["input_ids"].to(EMBED_FIRST_LAYER_DEVICE)
        attention_mask = inputs["attention_mask"].to(EMBED_FIRST_LAYER_DEVICE)
        batch_size, seq_len = input_ids.shape
        
        # 3. 初始化生成缓存
        past_key_values = DynamicCache()
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            current_seq_len = seq_len + step
            
            if step > 0:
                cache_position = torch.tensor([current_seq_len - 1], device=input_ids.device)
            else:
                cache_position = torch.arange(current_seq_len, device=input_ids.device)
            
            if step > 0:
                input_ids_step = generated_ids[:, -1:]
                position_ids_step = torch.tensor([[current_seq_len - 1]], device=input_ids.device).repeat(batch_size, 1)
            else:
                input_ids_step = generated_ids
                position_ids_step = None

            step1_output = self.embed_first_layer.forward(
                input_ids=input_ids_step,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position
            )
            
            step2_output = self.transformer_layers.forward(
                hidden_states=step1_output["hidden_states"],
                expanded_attn_mask=step1_output["expanded_attn_mask"],
                position_embeddings=step1_output["position_embeddings"],
                position_ids=step1_output["position_ids"],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position
            )
            
            step3_output = self.norm_lmhead_layer.forward(
                hidden_states=step2_output["hidden_states"]
            )
            
            next_token_ids = torch.argmax(step3_output["logits"][:, -1:, :], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            if (next_token_ids == self.tokenizer.eos_token_id).all():
                break
    
        decoded_texts = self.tokenizer.batch_decode(
            generated_ids.cpu(),
            skip_special_tokens=True
        )
        
        return decoded_texts

if __name__ == "__main__":
    pipeline = Qwen3Pipeline(
        model_path=MODEL_PATH,
        layers_safetensors_path=LAYERS_SAFETENSORS_PATH
    )
    
    # 测试输入
    input_texts = [
        "What is the capital of United States?",
        "你是谁？请介绍一下自己。"
    ]
    
    # 生成文本
    print("\n开始生成文本...")
    results = pipeline.generate(input_texts, max_new_tokens=30)
    
    # 输出结果
    print("\n=== 生成结果 ===")
    for i, text in enumerate(results):
        print(f"{i+1}. {text}")
    
    # 清理显存
    clean_memory()