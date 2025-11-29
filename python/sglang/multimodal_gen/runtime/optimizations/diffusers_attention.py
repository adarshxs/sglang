# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backends for diffusers.

Goal: Use SGLang's attention implementations (flash, sage, sdpa) with diffusers models.
If requested backend unavailable, fall back to diffusers default.
"""

import torch
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# =============================================================================
# Backend name mapping (CLI → canonical)
# =============================================================================

BACKEND_ALIASES = {
    "fa": "flash", "fa3": "flash", "fa4": "flash", "flash_attn": "flash",
    "sage_attn": "sage", "sage_attn_three": "sage",
    "torch_sdpa": "sdpa", "native": "sdpa",
}


def _resolve_backend(name: str) -> str:
    """Resolve alias to canonical name."""
    return BACKEND_ALIASES.get(name, name)


# =============================================================================
# Load SGLang backend implementations
# =============================================================================

def _load_impl(backend: str):
    """
    Load SGLang attention implementation.
    Returns (impl_instance, backend_name) or (None, None) if unavailable.
    """
    backend = _resolve_backend(backend)

    if backend == "flash":
        try:
            from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                FlashAttentionImpl,
            )
            impl = FlashAttentionImpl(
                num_heads=1, head_size=64, causal=False, softmax_scale=None
            )
            return impl, "flash"
        except ImportError:
            logger.debug("FlashAttention (sgl_kernel) not available")

    if backend == "sage":
        try:
            from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
                SageAttentionImpl,
            )
            impl = SageAttentionImpl(
                num_heads=1, head_size=64, causal=False, softmax_scale=None
            )
            return impl, "sage"
        except ImportError:
            logger.debug("SageAttention not available")

    if backend == "sdpa":
        from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
            SDPAImpl,
        )
        impl = SDPAImpl(
            num_heads=1, head_size=64, causal=False, softmax_scale=None
        )
        return impl, "sdpa"

    return None, None


def _load_best_impl():
    """Load best available: flash → sage → sdpa."""
    for backend in ["flash", "sage", "sdpa"]:
        impl, name = _load_impl(backend)
        if impl is not None:
            return impl, name
    return None, None


# =============================================================================
# Global state
# =============================================================================

_impl = None
_name = None
_patched = False


# =============================================================================
# Public API
# =============================================================================

def apply_sglang_attention(pipe, backend: str = "auto") -> bool:
    """
    Apply SGLang attention to diffusers pipeline.

    Args:
        pipe: Diffusers pipeline (unused, kept for interface compatibility)
        backend: "auto", "flash", "fa", "sage", "sdpa", etc.

    Returns:
        True if SGLang backend applied, False if using diffusers default.
    """
    global _impl, _name, _patched

    # Load implementation
    if backend == "auto":
        _impl, _name = _load_best_impl()
    else:
        _impl, _name = _load_impl(backend)
        if _impl is None:
            # Requested backend unavailable, try fallback
            logger.warning("%s not available, trying fallback", backend)
            _impl, _name = _load_best_impl()

    if _impl is None:
        logger.warning("No SGLang attention backend available, using diffusers default")
        return False

    # Patch diffusers dispatch
    if not _patched:
        _patch_diffusers()
        _patched = True

    logger.info("Using SGLang %s attention", _name)
    return True


def _patch_diffusers():
    """Patch diffusers' attention dispatch to use SGLang."""
    try:
        from diffusers.models import attention_dispatch
    except ImportError:
        logger.warning("diffusers.models.attention_dispatch not found")
        return

    # Save original
    if not hasattr(attention_dispatch, "_sglang_original"):
        attention_dispatch._sglang_original = attention_dispatch.dispatch_attention_fn

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
        """Route attention through SGLang backend."""
        global _impl

        if _impl is None:
            # Fallback to original
            return attention_dispatch._sglang_original(
                query, key, value, attn_mask, dropout_p, is_causal, scale, **kwargs
            )

        # Configure impl for this call
        _impl.causal = is_causal
        _impl.softmax_scale = scale if scale else query.shape[-1] ** -0.5

        # SGLang backends expect (batch, seq, heads, head_dim)
        # diffusers dispatch provides same shape
        return _impl.forward(query, key, value, attn_metadata=None)

    attention_dispatch.dispatch_attention_fn = sglang_dispatch


def get_current_backend() -> str | None:
    """Get name of active backend."""
    return _name


def reset():
    """Reset to diffusers default."""
    global _impl, _name, _patched

    try:
        from diffusers.models import attention_dispatch
        if hasattr(attention_dispatch, "_sglang_original"):
            attention_dispatch.dispatch_attention_fn = attention_dispatch._sglang_original
    except ImportError:
        pass

    _impl = None
    _name = None
    _patched = False
