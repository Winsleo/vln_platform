#!/usr/bin/env python3
"""
Minimal demo for unified processors.

This demo shows how to use LanguageProcessor and VisionLanguageProcessor
to prepare inputs and call model.generate in a unified way.

Note: Replace model ids/paths with your local checkpoints if needed.
"""

import sys
import os
# Add the multi_model_eval directory to Python path for standalone execution
# Since this file is in examples/ subdirectory, we need to go up one level
current_dir = os.path.dirname(os.path.abspath(__file__))
multi_model_eval_dir = os.path.dirname(current_dir)  # Go up from examples/ to multi_model_eval/
# Add both paths to ensure imports work
sys.path.insert(0, multi_model_eval_dir)
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from agents.processors.language_processor import LanguageProcessor
from agents.processors.vision_language_processor import Qwen25VLProcessor


def demo_language1() -> None:
    model_id = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to('cuda')
    processor = LanguageProcessor(tokenizer)

    prompt = 'Hello, briefly introduce vision-language models.'
    inputs = {
        'messages': prompt
    }
    model_inputs = processor.prepare_from_inputs(inputs)
    model_inputs.to('cuda')
    outputs = model.generate(**model_inputs, max_new_tokens=512)
    texts = processor.decode_trimmed(outputs, clean_up_tokenization_spaces=True)
    print('Language output:', texts[0])


def demo_language2() -> None:
    model_id = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

    processor = LanguageProcessor(tokenizer)

    prompt = "Briefly introduce the role of vision-language models (VL models) in two or three sentences."
    inputs = {
        'messages': prompt
    }
    model_inputs = processor.prepare_from_inputs(inputs, padding=True, add_generation_prompt=True)
    model_inputs.to('cuda')

    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)
    # Trim the prompt part for clean output
    texts = processor.decode_trimmed(outputs)
    print("distilgpt2 output:", texts[0].strip())


def demo_vision_language() -> None:
    import torch
    # Replace with your local Qwen2.5-VL path or use the hub id
    model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )
    hf_processor = AutoProcessor.from_pretrained(
        model_id, 
        padding_side='left'
    )
    processor = Qwen25VLProcessor(hf_processor)
    
    # Create a pure white PIL Image(224x224)
    white_image = Image.new('RGB', (224, 224), color='white')
    messages1 = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'What is the color of the image?'},
                # Just as a placeholder
                {'type': 'image'},
            ],
        },
    ]
    
    messages2 = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'What is in the image?'},
                # You can provide a URL or PIL.Image or path string
                {
                    'type': 'image',
                    'image': "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
            ],
        },
    ]
    inputs = [{'messages': messages1, 'images':white_image}, {'messages': messages2}]
    # Test single sample processing
    print("=== Single Sample Processing Test ===")
    single_inputs = processor.prepare_from_inputs(inputs[0])
    print(f"Single input shape: {single_inputs['input_ids'].shape}")
    
    # Test batch processing
    print("\n=== Batch Processing Test ===")
    batch_inputs = processor.prepare_from_inputs(inputs)
    print(f"Batch input shape: {batch_inputs['input_ids'].shape}")

    # generate
    single_generated_ids = model.generate(**single_inputs)
    batch_generated_ids = model.generate(**batch_inputs)
    single_texts = processor.decode_trimmed(single_generated_ids)
    batch_texts = processor.decode_trimmed(batch_generated_ids)
    print('single output:')
    print(single_texts[0])
    print()
    print('batch output:')
    print(batch_texts[0])


if __name__ == '__main__':
    # Run language demo (always available)
    demo_language1()
    demo_language2()
    # Run VL demo if you have the model downloaded/access
    demo_vision_language()
