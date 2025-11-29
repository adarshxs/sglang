# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.
Clean version with NO try/except blocks.
"""

from typing import Any
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import FlashAttentionImpl
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import SageAttentionImpl
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl

from diffusers.models import attention_dispatch

logger = init_logger(__name__)


# =============================================================================
# Backend fetching (no try/except)
# =============================================================================

def _get_flash_impl():
    return FlashAttentionImpl

def _get_sage_impl():
    return SageAttentionImpl

def _get_sdpa_impl():
    return SDPAImpl


# =============================================================================
# Backend registry
# =============================================================================

_BACKEND_MAP = {
    "fa": "flash",
    "fa3": "flash",
    "fa4": "flash",
    "flash": "flash",
    "flash_attn": "flash",
    "_flash_3": "flash",
    "_flash_3_hub": "flash",
    "sglang_fa": "flash",

    "sage": "sage",
    "sage_attn": "sage",
    "sage_attn_three": "sage",
    "sage_hub": "sage",
    "sglang_sage": "sage",

    "sdpa": "sdpa",
    "native": "sdpa",
    "torch_sdpa": "sdpa",
    "sglang_sdpa": "sdpa",
    "xformers": "sdpa",
    "flex": "sdpa",
}


# =============================================================================
# Backend selection
# =============================================================================

def _get_best_backend() -> tuple[str, Any]:
    """Choose best backend: flash → sage → sdpa."""
    return "flash", FlashAttentionImpl


def _get_backend_impl(backend: str) -> tuple[str, Any]:
    canonical = _BACKEND_MAP.get(backend, backend)

    if canonical == "flash":
        return "flash", FlashAttentionImpl

    if canonical == "sage":
        return "sage", SageAttentionImpl

    return "sdpa", SDPAImpl


# =============================================================================
# Global state
# =============================================================================

_attn_impl = None
_attn_name = None


# =============================================================================
# Main API
# =============================================================================

def apply_sglang_attention(pipe: Any, backend: str = "auto") -> None:
    global _attn_impl, _attn_name

    # Select implementation
    if backend == "auto":
        name, impl_cls = _get_best_backend()
    else:
        name, impl_cls = _get_backend_impl(backend)

    _attn_impl = impl_cls(
        num_heads=1,
        head_size=64,
        causal=False,
        softmax_scale=None,
    )
    _attn_name = name

    _patch_attention_dispatch()

    logger.info("Using SGLang %s attention backend", name)


# =============================================================================
# Patch diffusers attention dispatch
# =============================================================================

def _patch_attention_dispatch() -> None:
    """Override diffusers attention dispatch with SGLang backend."""

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

        # Update runtime properties
        _attn_impl.causal = is_causal
        _attn_impl.softmax_scale = scale or query.shape[-1] ** -0.5

        return _attn_impl.forward(query, key, value, attn_metadata=None)

    attention_dispatch.dispatch_attention_fn = sglang_dispatch


def get_current_backend() -> str | None:
    return _attn_name
