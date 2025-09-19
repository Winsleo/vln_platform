import torch


def move_to_device(obj, device):
    """
    Recursively moves all torch.Tensor objects in a nested data structure to the specified device.
    This operation is performed IN-PLACE for mutable containers like dictionaries and lists.
    Immutable containers (e.g., tuples) will be recreated.

    Args:
        obj: The Python object to process. This can be a tensor, a dictionary, a list, a tuple, or any nested combination of these.
        device: The target device to move the tensors to (e.g., 'cuda:0', 'cpu').

    Returns:
        The object with all tensors moved to the specified device. For mutable types
        like dicts and lists, this is the same object passed in. For immutable types
        like tuples, a new object is returned.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = move_to_device(v, device)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = move_to_device(v, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    return obj


def detect_best_attention_implementation():
    """
    Automatically detect the best available attention implementation with runtime compatibility checks.
    
    Returns:
        str: Best available attention implementation ('flash_attention_2', 'sdpa', or 'eager')
    
    Priority order:
    1. flash_attention_2 (best performance, requires Ampere GPU or newer)
    2. sdpa (good performance, PyTorch 2.0+, with runtime validation)
    3. eager (fallback, highest memory usage)
    """
    
    # Check if current GPU supports FlashAttention (Ampere or newer)
    def _is_ampere_or_newer():
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    
    # Try Flash Attention 2 with runtime compatibility check
    try:
        import flash_attn
        
        if hasattr(flash_attn, 'flash_attn_func') and _is_ampere_or_newer():
            # Test with small tensors
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                q = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                k = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                v = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                
                flash_attn.flash_attn_func(q, k, v)
                print("✓ Flash Attention 2 available and compatible")
                return 'flash_attention_2'
                
    except (ImportError, RuntimeError) as e:
        if "FlashAttention only supports Ampere GPUs" in str(e):
            capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            if capability:
                major, minor = capability
                print(f"⚠ GPU compute capability {major}.{minor} incompatible with FlashAttention (requires >= 8.0)")
        else:
            print(f"✗ Flash Attention unavailable: {type(e).__name__}")
    
    # Check PyTorch SDPA with runtime test and configure backends only when using SDPA
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        try:
            # Configure SDPA backends to avoid cutlass issues
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable cutlass
            
            # Test SDPA with small tensors
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                batch_size, num_heads, seq_len, head_dim = 1, 8, 32, 64
                
                q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print("✓ PyTorch SDPA available and compatible (cutlass disabled)")
                return 'sdpa'
                
        except RuntimeError as e:
            if "cutlassF" in str(e) or "no kernel found" in str(e):
                print(f"✗ SDPA cutlass kernel error: {e}")
            else:
                print(f"✗ SDPA runtime error: {e}")
        except Exception as e:
            print(f"✗ SDPA configuration failed: {e}")
    
    # Final fallback
    print("⚠ Using eager attention (may be slower)")
    return 'eager'
