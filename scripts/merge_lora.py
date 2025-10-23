#!/usr/bin/env python3
"""
简洁的单 LoRA 适配器合并脚本（支持多模态大模型）

特性：
- 仅合并一个 LoRA 适配器到基座模型
- 兼容多模态模型（例如含视觉塔/投影器），按 AutoModel* 加载
- 可选安全序列化与分片大小
- 保存模型、分词器与（若存在）生成配置
"""

import argparse
import logging
import os
import sys
import torch
import importlib
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
)
import transformers
from peft import PeftModel, PeftConfig

# Add project root to Python path to allow for custom module imports
# This makes the script runnable from anywhere and allows finding custom modules like StreamVLN
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
streamvln_root = os.path.join(project_root, "StreamVLN")
streamvln_model_root = os.path.join(streamvln_root, "streamvln")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, streamvln_root)
    sys.path.insert(0, streamvln_model_root)
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_single_lora")


def _resolve_model_class(model_class_name: str):
    """从 transformers 或自定义模块动态解析模型类名。

    支持：
    1. 从自定义模块路径加载 (e.g., "my_models.custom_arch:MyModelClass")
    2. 直接从 transformers 库解析 (e.g., "AutoModelForCausalLM")
    """
    # 优先尝试作为自定义模块路径 "path.to.module:ClassName" 处理
    if ":" in model_class_name:
        module_name, class_name = model_class_name.rsplit(":", 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError as e:
            raise ValueError(
                f"Could not import module '{module_name}' to load custom model class '{class_name}'. "
                f"Ensure the module is in your PYTHONPATH."
            ) from e
        except AttributeError as e:
            raise ValueError(
                f"Class '{class_name}' not found in module '{module_name}'."
            ) from e

    # 若非自定义格式，则回退到从 transformers 库解析
    try:
        return getattr(transformers, model_class_name)
    except AttributeError as e:
        raise ValueError(
            f"Cannot resolve model class '{model_class_name}' from transformers. "
            f"If it's a custom model, use the format 'path.to.module:ClassName'."
        ) from e


def load_base_model(
    model_name_or_path: str,
    model_class: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
):
    dtype = (
        torch.float16 if torch_dtype == "auto" and torch.cuda.is_available() else
        getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float32
    )
    
    # 预加载配置，以检查并修复潜在的嵌套字典问题
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code
    )
    
    # 修复：如果 decoder 是一个字典而非配置对象，则进行转换
    # 这可以防止在 GenerationConfig 初始化时出现 'dict' object has no attribute 'to_dict' 错误
    if hasattr(config, "decoder") and isinstance(config.decoder, dict):
        from transformers import PretrainedConfig
        logger.info("Decoder config is a dict, converting to PretrainedConfig object.")
        config.decoder = PretrainedConfig.from_dict(config.decoder)

    kwargs = {"device_map": device_map, "torch_dtype": dtype, "trust_remote_code": trust_remote_code}

    model_cls = _resolve_model_class(model_class)
    # 使用修复后的 config 加载模型
    model = model_cls.from_pretrained(model_name_or_path, config=config, **kwargs)
    return model


def merge_single_lora(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    model_class: str,
    non_lora_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
):
    logger.info(f"Loading base model: {base_model_path}")
    base_model = load_base_model(
        base_model_path,
        model_class=model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    logger.info(f"Loading LoRA adapter: {lora_adapter_path}")
    peft_cfg = PeftConfig.from_pretrained(lora_adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    logger.info(
        f"LoRA loaded (r={getattr(peft_cfg,'r','N/A')}, alpha={getattr(peft_cfg,'lora_alpha','N/A')},"
        f" targets={getattr(peft_cfg,'target_modules','N/A')})"
    )

    logger.info("Merging LoRA into base model...")
    merged = peft_model.merge_and_unload()

    # 可选：合并非 LoRA 的可训练权重（训练时单独保存的权重，如 lm_head 等）
    if non_lora_path is not None:
        if os.path.exists(non_lora_path):
            logger.info(f"Loading non-LoRA trainables: {non_lora_path}")
            # 使用 weights_only=True 增强安全性，防止执行任意代码
            non_lora_state = torch.load(non_lora_path, map_location="cpu", weights_only=True)

            # 构建 name -> tensor 引用表，覆盖参数与缓冲区
            param_map = dict(merged.named_parameters())
            buffer_map = dict(merged.named_buffers())

            loaded_cnt = 0
            missing = []
            shape_mismatch = []

            with torch.no_grad():
                for name, tensor in non_lora_state.items():
                    # 简化查找逻辑：在参数或缓冲区中查找目标张量
                    target = param_map.get(name) or buffer_map.get(name)

                    if target is None:
                        missing.append(name)
                        continue

                    if target.shape != tensor.shape:
                        shape_mismatch.append((name, tuple(tensor.shape), tuple(target.shape)))
                        continue

                    # In-place copy，确保设备与数据类型匹配
                    casted = tensor.to(dtype=target.dtype, device=target.device, non_blocking=True)
                    target.copy_(casted)
                    loaded_cnt += 1

            if missing:
                logger.warning(f"Non-LoRA keys not found in model (first 10): {missing[:10]} (total={len(missing)})")
            if shape_mismatch:
                logger.warning(
                    "Non-LoRA keys with shape mismatch (first 5): " +
                    ", ".join([f"{n}: src{src} != dst{dst}" for n, src, dst in shape_mismatch[:5]]) +
                    f" (total={len(shape_mismatch)})"
                )
            logger.info(f"Applied non-LoRA trainables: {loaded_cnt} tensors")
        else:
            logger.warning(f"non_lora_path not found, skip applying: {non_lora_path}")

    logger.info(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path, safe_serialization=safe_serialization, max_shard_size=max_shard_size)

    # 优先保存 AutoProcessor；若不存在再回退保存 AutoTokenizer
    saved_processor = False
    try:
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_path)
        saved_processor = True
    except Exception as e:
        logger.warning(f"Processor not saved: {e}")

    if not saved_processor:
        try:
            tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
            tok.save_pretrained(output_path)
        except Exception as e:
            logger.warning(f"Tokenizer not saved: {e}")

    try:
        cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        cfg.save_pretrained(output_path)
    except Exception as e:
        logger.warning(f"Config not saved: {e}")

    try:
        if hasattr(merged, "generation_config") and merged.generation_config is not None:
            merged.generation_config.save_pretrained(output_path)
    except Exception as e:
        logger.warning(f"Generation config not saved: {e}")

    logger.info("Done.")


def main():
    p = argparse.ArgumentParser(description="Merge a single LoRA adapter into a base model")
    p.add_argument("--base_model", '-b', required=True, help="Base model path or repo id")
    p.add_argument("--lora_adapter", '-l', required=True, help="LoRA adapter path")
    p.add_argument("--output_path", '-o', required=True, help="Output path for merged model")
    p.add_argument("--model_class", '-m', required=True, help="Model class name. For transformers models: 'AutoModelForCausalLM'. For custom models: 'path.to.module:ClassName'")
    p.add_argument("--non_lora_path", '-n', default=None, help="Path to non-LoRA trainables bin (e.g., non_lora_trainables.bin)")
    p.add_argument("--device_map", '-d', default="auto", help="Device map for loading")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Torch dtype")
    p.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    p.add_argument("--safe_serialization", action="store_true", default=True, help="Use safetensors")
    p.add_argument("--max_shard_size", default="5GB", help="Max shard size when saving")
    args = p.parse_args()

    merge_single_lora(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_path,
        model_class=args.model_class,
        non_lora_path=args.non_lora_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()


