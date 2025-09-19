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
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
)
import transformers
from peft import PeftModel, PeftConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_single_lora")


def _resolve_model_class(model_class_name: str):
    """从 transformers 动态解析模型类名（不做回退）。

    支持：如 "AutoModelForCausalLM"、"Qwen2_5_VLForConditionalGeneration" 等，需要当前 transformers 版本已导出该符号。
    """
    try:
        return getattr(transformers, model_class_name)
    except AttributeError as e:
        raise ValueError(
            f"Cannot resolve model class '{model_class_name}' from transformers. "
            f"Ensure the class name is correct and your transformers version supports it."
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
    kwargs = {"device_map": device_map, "torch_dtype": dtype, "trust_remote_code": trust_remote_code}

    model_cls = _resolve_model_class(model_class)
    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    return model


def merge_single_lora(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    model_class: str,
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
    p.add_argument("--model_class", '-m', required=True, help="Transformers model class name, e.g., AutoModelForCausalLM or Qwen2_5_VLForConditionalGeneration")
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
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()


