# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.

Simple flow:
1. Try SGLang backend (sglang_fa, sglang_sage) via dispatch patch
2. Fall back to SDPA
"""

from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# =============================================================================
# SGLang Attention Implementations
# =============================================================================

def _get_sglang_flash_attn():
    """Get SGLang's FlashAttention from sgl_kernel."""
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func
    except ImportError:
        return None


def _get_sglang_sage_attn():
    """Get SageAttention."""
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
    SGLang FlashAttention using sgl_kernel.
    Input: (batch, seq, heads, head_dim)
    """
    flash_fn = _get_sglang_flash_attn()
    if flash_fn is None:
        raise ImportError("sgl_kernel.flash_attn not available")

    batch_size, seq_len, num_heads, head_dim = query.shape

    # Reshape for varlen interface
    q = query.reshape(-1, num_heads, head_dim)
    k = key.reshape(-1, num_heads, head_dim)
    v = value.reshape(-1, num_heads, head_dim)

    # Cumulative sequence lengths
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=query.device
    )

    output = flash_fn(
        q, k, v,
        cu_seqlens, cu_seqlens,
        seq_len, seq_len,
        softmax_scale=head_dim ** -0.5,
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
    SageAttention wrapper.
    Input: (batch, seq, heads, head_dim)
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


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    PyTorch SDPA fallback.
    Input: (batch, seq, heads, head_dim)
    """
    # SDPA expects (batch, heads, seq, head_dim)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=query.shape[-1] ** -0.5,
    )

    return output.transpose(1, 2)


# =============================================================================
# Main API
# =============================================================================

# Backend name mapping
_BACKEND_MAP = {
    # SGLang backends
    "sglang_fa": "sglang_fa",
    "sglang_sage": "sglang_sage",
    # CLI aliases -> SGLang
    "fa": "sglang_fa",
    "fa3": "sglang_fa",
    "fa4": "sglang_fa",
    "flash_attn": "sglang_fa",
    "flash": "sglang_fa",
    "sage_attn": "sglang_sage",
    "sage": "sglang_sage",
    # SDPA
    "sdpa": "sdpa",
    "native": "sdpa",
    "torch_sdpa": "sdpa",
}


def apply_sglang_attention(pipe: Any, backend: str = "auto") -> None:
    """
    Apply SGLang attention backend to diffusers pipeline.

    Flow:
    1. Try SGLang backend (patches diffusers dispatch)
    2. Fall back to SDPA

    Args:
        pipe: Diffusers pipeline
        backend: "auto", "sglang_fa", "sglang_sage", "fa", "sage", "sdpa", etc.
    """
    # Map backend name
    mapped = _BACKEND_MAP.get(backend, backend)

    # Auto-select
    if mapped == "auto" or backend == "auto":
        if _get_sglang_flash_attn() is not None:
            mapped = "sglang_fa"
        elif _get_sglang_sage_attn() is not None:
            mapped = "sglang_sage"
        else:
            mapped = "sdpa"
        logger.info("Auto-selected attention backend: %s", mapped)

    # Get attention function
    if mapped == "sglang_fa":
        if _get_sglang_flash_attn() is None:
            logger.warning("sgl_kernel.flash_attn not available, using SDPA")
            mapped = "sdpa"
    elif mapped == "sglang_sage":
        if _get_sglang_sage_attn() is None:
            logger.warning("sageattention not available, using SDPA")
            mapped = "sdpa"

    attn_fn = {
        "sglang_fa": sglang_flash_attention,
        "sglang_sage": sglang_sage_attention,
        "sdpa": sdpa_attention,
    }.get(mapped, sdpa_attention)

    # Patch diffusers dispatch
    _patch_attention_dispatch(attn_fn, mapped)


def _patch_attention_dispatch(attn_fn, backend_name: str) -> None:
    """Patch diffusers' attention dispatch to use our function."""
    try:
        from diffusers.models import attention_dispatch

        original = getattr(
            attention_dispatch, "_original_dispatch", None
        ) or attention_dispatch.dispatch_attention_fn
        attention_dispatch._original_dispatch = original

        def patched_dispatch(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float = None,
            **kwargs,
        ) -> torch.Tensor:
            return attn_fn(query, key, value, is_causal=is_causal)

        attention_dispatch.dispatch_attention_fn = patched_dispatch
        logger.info("Using %s attention backend", backend_name)

    except Exception as e:
        logger.warning("Could not patch attention dispatch: %s", e)
