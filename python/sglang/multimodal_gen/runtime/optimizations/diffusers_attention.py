# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.

Uses SGLang's actual attention backend implementations directly.
"""

from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# =============================================================================
# Get SGLang Backend Implementations
# =============================================================================

def _get_flash_impl():
    """Get SGLang FlashAttention implementation."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
        )
        return FlashAttentionImpl
    except ImportError:
        return None


def _get_sage_impl():
    """Get SGLang SageAttention implementation."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
            SageAttentionImpl,
        )
        return SageAttentionImpl
    except ImportError:
        return None


def _get_sdpa_impl():
    """Get SGLang SDPA implementation."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
            SDPAImpl,
        )
        return SDPAImpl
    except ImportError:
        return None


# =============================================================================
# Backend Selection
# =============================================================================

# User input → canonical backend type
_BACKEND_MAP = {
    # Flash variants
    "fa": "flash",
    "fa3": "flash",
    "fa4": "flash",
    "flash_attn": "flash",
    "flash": "flash",
    "sglang_fa": "flash",
    "_flash_3": "flash",
    # Sage variants  
    "sage": "sage",
    "sage_attn": "sage",
    "sglang_sage": "sage",
    # SDPA variants
    "sdpa": "sdpa",
    "native": "sdpa",
    "torch_sdpa": "sdpa",
    "sglang_sdpa": "sdpa",
}


def _get_best_backend() -> tuple[str, Any]:
    """Auto-select best available backend. Returns (name, impl_class)."""
    # Try flash first
    flash_impl = _get_flash_impl()
    if flash_impl is not None:
        return "flash", flash_impl
    
    # Try sage
    sage_impl = _get_sage_impl()
    if sage_impl is not None:
        return "sage", sage_impl
    
    # Fall back to SDPA
    sdpa_impl = _get_sdpa_impl()
    return "sdpa", sdpa_impl


def _get_backend_impl(backend: str) -> tuple[str, Any]:
    """Get implementation for specified backend. Returns (name, impl_class)."""
    canonical = _BACKEND_MAP.get(backend, backend)
    
    if canonical == "flash":
        impl = _get_flash_impl()
        if impl is not None:
            return "flash", impl
        logger.warning("FlashAttention not available, trying sage")
        canonical = "sage"
    
    if canonical == "sage":
        impl = _get_sage_impl()
        if impl is not None:
            return "sage", impl
        logger.warning("SageAttention not available, using SDPA")
        canonical = "sdpa"
    
    # SDPA fallback
    return "sdpa", _get_sdpa_impl()


# =============================================================================
# Main API
# =============================================================================

# Global attention implementation instance
_attn_impl = None
_attn_name = None


def apply_sglang_attention(pipe: Any, backend: str = "auto") -> None:
    """
    Apply SGLang attention backend to diffusers pipeline.

    Uses SGLang's actual attention implementations:
    - FlashAttentionImpl (sgl_kernel.flash_attn)
    - SageAttentionImpl (sageattention)
    - SDPAImpl (torch.nn.functional.scaled_dot_product_attention)

    Args:
        pipe: Diffusers pipeline
        backend: "auto", "fa", "flash", "sage", "sdpa", etc.
    """
    global _attn_impl, _attn_name
    
    # Get backend implementation
    if backend == "auto":
        name, impl_cls = _get_best_backend()
    else:
        name, impl_cls = _get_backend_impl(backend)
    
    if impl_cls is None:
        logger.error("No attention backend available!")
        return
    
    # Create implementation instance
    # Default params work for most diffusers models
    _attn_impl = impl_cls(
        num_heads=1,  # Not used in forward
        head_size=64,  # Not used in forward
        causal=False,
        softmax_scale=None,  # Will be computed from head_dim
    )
    _attn_name = name
    
    # Patch diffusers dispatch
    _patch_attention_dispatch()
    logger.info("Using SGLang %s attention backend", name)


def _patch_attention_dispatch() -> None:
    """Patch diffusers' attention dispatch to use SGLang backend."""
    try:
        from diffusers.models import attention_dispatch
        
        # Store original
        if not hasattr(attention_dispatch, "_original_dispatch"):
            attention_dispatch._original_dispatch = attention_dispatch.dispatch_attention_fn
        
        def sglang_dispatch(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float = None,
            **kwargs,
        ) -> torch.Tensor:
            global _attn_impl
            
            # Update causal flag
            _attn_impl.causal = is_causal
            
            # Compute softmax scale if not provided
            if scale is not None:
                _attn_impl.softmax_scale = scale
            else:
                _attn_impl.softmax_scale = query.shape[-1] ** -0.5
            
            # Call SGLang implementation
            return _attn_impl.forward(query, key, value, attn_metadata=None)
        
        attention_dispatch.dispatch_attention_fn = sglang_dispatch
        
    except ImportError:
        logger.warning("diffusers.models.attention_dispatch not available")
    except Exception as e:
        logger.warning("Failed to patch attention dispatch: %s", e)


def get_current_backend() -> str | None:
    """Get name of currently active backend."""
    return _attn_name
