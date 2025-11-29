# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.

This module provides two approaches:
1. Use diffusers' native attention dispatcher API (recommended)
2. Use SGLang's attention backends directly (bypassing parallelism)

See: https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
"""

from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Map SGLang/CLI backend names to diffusers backend names
_BACKEND_NAME_MAP = {
    # SGLang CLI names
    "fa": "flash",
    "fa3": "_flash_3",
    "fa4": "_flash_3",
    "flash_attn": "flash",
    "torch_sdpa": "native",
    "sdpa": "native",
    "sage_attn": "sage",
    "sage_attn_three": "_sage_qk_int8_pv_fp8_cuda",
    "sage_attn_3": "_sage_qk_int8_pv_fp8_cuda",
    # Direct diffusers names (pass through)
    "flash": "flash",
    "flash_hub": "flash_hub",
    "_flash_3": "_flash_3",
    "_flash_3_hub": "_flash_3_hub",
    "sage": "sage",
    "sage_hub": "sage_hub",
    "native": "native",
    "xformers": "xformers",
    "flex": "flex",
    "auto": "auto",
    # SGLang direct backends (bypass diffusers dispatcher)
    "sglang_fa": "sglang_fa",
    "sglang_sage": "sglang_sage",
    "sglang_sdpa": "sglang_sdpa",
}


# =============================================================================
# SGLang Backend Wrappers (bypass parallelism, use raw attention implementations)
# =============================================================================

def _get_sglang_flash_attn():
    """Get SGLang's FlashAttention implementation (without parallelism)."""
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func
    except ImportError:
        return None


def _get_sglang_sage_attn():
    """Get SGLang's SageAttention implementation."""
    try:
        from sageattention import sageattn
        return sageattn
    except ImportError:
        return None


def sglang_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    SGLang FlashAttention wrapper for diffusers.
    
    Bypasses SGLang's parallelism infrastructure, uses raw flash_attn_varlen_func.
    Input shape: (batch, seq, heads, head_dim)
    """
    flash_fn = _get_sglang_flash_attn()
    if flash_fn is None:
        raise ImportError("sgl_kernel.flash_attn not available")
    
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # Reshape to (batch*seq, heads, head_dim) for varlen interface
    q = query.reshape(-1, num_heads, head_dim)
    k = key.reshape(-1, num_heads, head_dim)
    v = value.reshape(-1, num_heads, head_dim)
    
    # Create cumulative sequence lengths
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=query.device
    )
    
    softmax_scale = head_dim ** -0.5
    
    output = flash_fn(
        q, k, v,
        cu_seqlens, cu_seqlens,
        seq_len, seq_len,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )
    
    return output.reshape(batch_size, seq_len, num_heads, head_dim)


def sglang_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    SGLang SageAttention wrapper for diffusers.
    
    Input shape: (batch, seq, heads, head_dim)
    """
    sage_fn = _get_sglang_sage_attn()
    if sage_fn is None:
        raise ImportError("sageattention not available")
    
    # SageAttention expects (batch, heads, seq, head_dim)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    
    output = sage_fn(q, k, v, is_causal=is_causal, smooth_k=True)
    
    return output.transpose(1, 2)


def sglang_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    SGLang SDPA wrapper (same as PyTorch native, for consistency).
    
    Input shape: (batch, seq, heads, head_dim)
    """
    import torch.nn.functional as F
    
    # SDPA expects (batch, heads, seq, head_dim)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    
    softmax_scale = query.shape[-1] ** -0.5
    
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=softmax_scale,
    )
    
    return output.transpose(1, 2)


# Registry of SGLang attention functions
SGLANG_ATTENTION_FUNCTIONS = {
    "sglang_fa": sglang_flash_attention,
    "sglang_sage": sglang_sage_attention,
    "sglang_sdpa": sglang_sdpa_attention,
}


def register_sglang_backends_with_diffusers() -> bool:
    """
    Register SGLang attention backends with diffusers' attention dispatcher.
    
    Note: This may not work with all diffusers versions as the registry
    has internal validation. If registration fails, use diffusers' built-in
    backends (flash, sage, native) instead.
    
    Returns:
        True if registration succeeded, False otherwise.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
        
        # Try to register each SGLang backend
        registered = []
        for name, fn in SGLANG_ATTENTION_FUNCTIONS.items():
            try:
                if hasattr(_AttentionBackendRegistry, "_backends"):
                    if name not in _AttentionBackendRegistry._backends:
                        _AttentionBackendRegistry._backends[name] = fn
                        registered.append(name)
                elif hasattr(_AttentionBackendRegistry, "register"):
                    _AttentionBackendRegistry.register(name, fn)
                    registered.append(name)
            except Exception as e:
                logger.debug("Could not register '%s': %s", name, e)
        
        if registered:
            logger.info("Registered SGLang backends: %s", ", ".join(registered))
            return True
        else:
            logger.debug("No SGLang backends were registered (diffusers may validate backends)")
            return False
        
    except ImportError:
        logger.debug("diffusers attention dispatcher not available")
        return False
    except Exception as e:
        logger.debug("Failed to register SGLang backends: %s", e)
        return False

# Backends to try in order of preference for "auto"
_AUTO_BACKENDS = [
    "_flash_3",  # FlashAttention 3 (Hopper)
    "flash",     # FlashAttention 2
    "sage",      # SageAttention
    "native",    # PyTorch SDPA (always available)
]


def get_available_attention_backends() -> list[str]:
    """Get list of attention backends available in diffusers."""
    available = ["native"]  # Always available

    try:
        # Check if flash attention is available
        import flash_attn  # noqa: F401
        available.extend(["flash", "_flash_3"])
    except ImportError:
        pass

    try:
        from sageattention import sageattn  # noqa: F401
        available.append("sage")
    except ImportError:
        pass

    try:
        import xformers  # noqa: F401
        available.append("xformers")
    except ImportError:
        pass

    return available


def _patch_diffusers_attention_dispatch(backend: str) -> bool:
    """
    Directly patch diffusers' attention dispatch to use SGLang's attention.
    
    This bypasses set_attention_backend() validation by patching the dispatch
    function itself.
    """
    try:
        from diffusers.models import attention_dispatch
        
        # Get the SGLang attention function
        if backend == "sglang_fa":
            attn_fn = sglang_flash_attention
        elif backend == "sglang_sage":
            attn_fn = sglang_sage_attention
        elif backend == "sglang_sdpa":
            attn_fn = sglang_sdpa_attention
        else:
            return False
        
        # Store original dispatch function
        original_dispatch = getattr(attention_dispatch, "_original_dispatch_attention_fn", None)
        if original_dispatch is None:
            original_dispatch = attention_dispatch.dispatch_attention_fn
            attention_dispatch._original_dispatch_attention_fn = original_dispatch
        
        # Create wrapper that uses SGLang attention
        def sglang_dispatch_attention_fn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float = None,
            **kwargs,
        ) -> torch.Tensor:
            try:
                # Use SGLang attention
                return attn_fn(query, key, value, is_causal=is_causal)
            except Exception:
                # Fall back to original dispatch on any error
                return original_dispatch(
                    query, key, value, attn_mask, dropout_p, is_causal, scale, **kwargs
                )
        
        # Patch the dispatch function
        attention_dispatch.dispatch_attention_fn = sglang_dispatch_attention_fn
        logger.info("Patched diffusers attention dispatch with SGLang %s", backend)
        return True
        
    except ImportError:
        logger.debug("diffusers.models.attention_dispatch not available")
        return False
    except Exception as e:
        logger.debug("Failed to patch attention dispatch: %s", e)
        return False


def apply_sglang_attention(
    pipe: Any,
    backend: str = "auto",
) -> None:
    """
    Apply optimized attention backend to a diffusers pipeline.

    For SGLang backends (sglang_fa, sglang_sage), directly patches diffusers'
    attention dispatch function to use SGLang's implementations.

    For diffusers backends (flash, sage, native), uses set_attention_backend().

    Args:
        pipe: A diffusers pipeline
        backend: Attention backend. Options:
            - "auto": Auto-select best available
            - "sglang_fa": SGLang FlashAttention (RECOMMENDED - uses sgl_kernel)
            - "sglang_sage": SGLang SageAttention
            - "sglang_sdpa": SGLang SDPA wrapper
            - "fa", "flash", "_flash_3": diffusers FlashAttention
            - "sage": diffusers SageAttention
            - "native": PyTorch SDPA
    """
    # Map to backend name
    mapped_backend = _BACKEND_NAME_MAP.get(backend, backend)

    # Handle auto selection - prefer SGLang backends
    if mapped_backend == "auto":
        if _get_sglang_flash_attn() is not None:
            mapped_backend = "sglang_fa"
            logger.info("Auto-selected SGLang FlashAttention backend")
        elif _get_sglang_sage_attn() is not None:
            mapped_backend = "sglang_sage"
            logger.info("Auto-selected SGLang SageAttention backend")
        else:
            mapped_backend = "native"
            logger.info("Auto-selected native SDPA backend")

    # For SGLang backends, patch the dispatch function directly
    if mapped_backend.startswith("sglang_"):
        success = _patch_diffusers_attention_dispatch(mapped_backend)
        if success:
            return  # Successfully patched, we're done
        else:
            # Fall back to equivalent diffusers backend
            fallback = {
                "sglang_fa": "flash",
                "sglang_sage": "sage",
                "sglang_sdpa": "native",
            }
            mapped_backend = fallback.get(mapped_backend, "native")
            logger.warning(
                "Could not patch SGLang attention, falling back to diffusers '%s'",
                mapped_backend,
            )

    # For diffusers backends, use set_attention_backend with fallback
    models = []
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models.append(("transformer", pipe.transformer))
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models.append(("unet", pipe.unet))

    if not models:
        logger.debug("No transformer or unet found in pipeline")
        return

    for name, model in models:
        if not hasattr(model, "set_attention_backend"):
            logger.info("%s does not support set_attention_backend", name)
            continue

        # Try backends with fallback to native
        backends_to_try = [mapped_backend]
        if mapped_backend != "native":
            backends_to_try.append("native")

        for try_backend in backends_to_try:
            try:
                model.set_attention_backend(try_backend)
                logger.info("Set attention backend '%s' on %s", try_backend, name)
                break
            except Exception as e:
                logger.debug("Backend '%s' failed: %s", try_backend, e)
                continue


def reset_attention_backend(pipe: Any) -> None:
    """Reset attention backend to default for all models in pipeline."""
    models = []
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models.append(("transformer", pipe.transformer))
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models.append(("unet", pipe.unet))

    for name, model in models:
        try:
            if hasattr(model, "reset_attention_backend"):
                model.reset_attention_backend()
                logger.debug("Reset attention backend on %s", name)
        except Exception as e:
            logger.warning("Could not reset attention backend on %s: %s", name, e)


# Legacy aliases for compatibility
class SGLangAttnProcessor:
    """Legacy class - use apply_sglang_attention() instead."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "SGLangAttnProcessor is deprecated. "
            "Use apply_sglang_attention() which uses diffusers' native API."
        )


SGLangAttnProcessor2_0 = SGLangAttnProcessor
