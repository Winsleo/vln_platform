#!/usr/bin/env python3
"""
Model download utility

This script downloads required models from Hugging Face Hub to local storage.

Features:
- Batch download multiple Hugging Face models
- Resume partial downloads
- Basic integrity checks (presence of config.json, rough size)
- Optional file filtering to save disk space

Usage:
    python3 download_models.py
"""

import os
import sys


def check_network():
    """
    Check basic network connectivity to Hugging Face Hub.

    Returns:
        bool: True if reachable, False otherwise.

    Note:
        Performs a simple HTTP GET to https://huggingface.co with a 10s timeout.
    """
    try:
        import requests
        print("🔍 Checking network connectivity ...")
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✅ Network looks OK")
            return True
        else:
            print(f"❌ Network issue. HTTP status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network check failed: {e}")
        return False


def download_model(model_name, model_description="model", ignore_patterns=None):
    """
    Download a single model snapshot from Hugging Face Hub.

    Args:
        model_name (str): HF repo id, e.g. "google/siglip-so400m-patch14-384"
        model_description (str): Friendly name for logs
        ignore_patterns (list): File patterns to ignore, e.g. ["*.safetensors.index.json"]

    Returns:
        tuple: (success: bool, local_dir: str)
    """
    try:
        from huggingface_hub import snapshot_download
        
        # Local directory under ./checkpoints; keeps repo structure (org/model)
        local_dir = f"{os.getcwd()}/checkpoints/{model_name}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"\n📥 Downloading {model_description}: {model_name}")
        print(f"💾 Save to: {local_dir}")
        
        # Prepare download args
        download_kwargs = {
            "repo_id": model_name,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True
        }
        
        # Optional ignore patterns
        if ignore_patterns:
            download_kwargs["ignore_patterns"] = ignore_patterns
        
        # Download full repo snapshot
        snapshot_download(**download_kwargs)
        
        print(f"✅ {model_description} downloaded: {local_dir}")
        return True, local_dir
        
    except Exception as e:
        print(f"❌ {model_description} download failed: {e}")
        return False, None


def verify_model(model_path, model_description):
    """
    Basic verification for a downloaded model.

    Args:
        model_path (str): Local path
        model_description (str): Friendly name for logs

    Returns:
        bool: True if OK, False otherwise
    """
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        print(f"✅ {model_description}: config.json found")
        
        # Total size info (rough)
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        size_mb = total_size / (1024 * 1024)
        print(f"   📊 Total size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ {model_description}: config.json not found ({model_path})")
        return False


def verify_downloads(downloaded_models):
    """
    Verify all downloaded models.

    Args:
        downloaded_models (list): List of tuples (model_path, model_description)

    Returns:
        bool: True if all verified, else False
    """
    print("\n🔍 Verifying downloaded models ...")
    
    all_valid = True
    for model_path, model_description in downloaded_models:
        if not verify_model(model_path, model_description):
            all_valid = False
    
    return all_valid


def download_single_model(model_name, model_description=None, ignore_patterns=None):
    """
    Convenience function to download a single model.

    Args:
        model_name (str): HF repo id
        model_description (str): Friendly name (defaults to model_name)
        ignore_patterns (list): File patterns to ignore

    Returns:
        bool: True on success
    """
    if model_description is None:
        model_description = model_name
    
    print("🚀 Single-model download")
    print("=" * 50)
    
    # Check dependency
    try:
        import huggingface_hub
    except ImportError:
        print("❌ Please install huggingface_hub: pip install huggingface_hub")
        return False
    
    # Network check
    if not check_network():
        print("\n💡 Tips:")
        print("1. Check your network connection")
        print("2. Configure proxy (if needed)")
        print("3. Try a VPN (if applicable)")
        return False
    
    # Download
    success, local_dir = download_model(model_name, model_description, ignore_patterns)
    
    if success:
        # Verify
        if verify_model(local_dir, model_description):
            print("✅ Verification passed!")
        print(f"\n🎉 Model {model_name} downloaded!")
        return True
    else:
        print(f"\n❌ Failed to download model {model_name}!")
        return False


def download_models(models_params):
    """
    Batch download multiple models.

    Args:
        models_params (list): List of dicts with keys:
            - name (str): HF repo id, e.g. "organization/model-name"
            - description (str): Description for logging
            - ignore_patterns (list or None): File patterns to ignore during download

    Returns:
        bool: True if all models downloaded successfully, else False

    Example:
        models_params = [
            {
                "name": "google/siglip-so400m-patch14-384",
                "description": "SigLIP vision model",
                "ignore_patterns": ["*.bin"]  # keep only safetensors
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "Dialogue model",
                "ignore_patterns": None
            }
        ]
        success = download_models(models_params)
    """
    print("🚀 StreamVLN Model Downloader")
    print("=" * 50)
    
    # Check dependency
    try:
        import huggingface_hub
    except ImportError:
        print("❌ Please install huggingface_hub: pip install huggingface_hub")
        return False
    
    # Network check
    if not check_network():
        print("\n💡 Tips:")
        print("1. Check your network connection")
        print("2. Configure proxy (if needed)")
        print("3. Try a VPN (if applicable)")
        return False
    
    # Download models
    downloaded_models = []  # [(local_dir, description), ...]
    success_count = 0
    
    # Iterate and download
    for model_config in models_params:
        print("\n" + "="*50)
        success, local_dir = download_model(
            model_name=model_config["name"],
            model_description=model_config["description"],
            ignore_patterns=model_config["ignore_patterns"]
        )
        
        if success:
            success_count += 1
            downloaded_models.append((local_dir, model_config["description"]))
    
    print("\n" + "="*50)
    
    # Verify integrity
    if downloaded_models and verify_downloads(downloaded_models):
        print("✅ All models verified!")
    
    # Summary
    total_models = len(models_params)
    print(f"\n🎉 Done! Downloaded {success_count}/{total_models} models")
    
    if success_count == total_models:
        print("\n✅ All models downloaded successfully!")
        print("You can now run StreamVLN offline using local weights.")
        print("\nNext steps:")
        print("1. Set envs: export DATA_PATH=$PWD/data MODEL_PATH=$PWD/checkpoints")
        print("2. Start container: docker compose up")
        print("3. Run evaluation: sh scripts/streamvln_eval_multi_gpu.sh")
    else:
        print("\n⚠️ Some models failed to download. Please check network/proxy and retry.")
        
    return success_count == total_models


if __name__ == "__main__":
    # Define models to download
    models_to_download = [
        {
            "name": "google/siglip-so400m-patch14-384",
            "description": "SigLIP vision model",
            "ignore_patterns": None
        },
        {
            "name": "mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln",
            "description": "StreamVLN main model",
            "ignore_patterns": None  # e.g., ["*.safetensors.index.json"] to save space
        },
        {
            "name": "lmms-lab/LLaVA-Video-7B-Qwen2",
            "description": "LLaVA-Video model",
            "ignore_patterns": None
        },
        {
            'name': "a8cheng/navila-llama3-8b-8f",
            'description': 'Navila model',
            'ignore_patterns': None
        },
        {
            'name': "Qwen/Qwen2.5-VL-3B-Instruct",
            'description': 'Qwen2.5-VL-3B-Instruct',
            'ignore_patterns': None
        },
        
    ]
    success = download_models(models_to_download)
    
    sys.exit(0 if success else 1)
