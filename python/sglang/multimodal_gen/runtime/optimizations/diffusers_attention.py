# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.

Flow:
1. Query model's supported backends
2. If requested backend is supported, use SGLang's implementation
3. Fall back to SDPA
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
    """SGLang FlashAttention. Input: (batch, seq, heads, head_dim)"""
    flash_fn = _get_sglang_flash_attn()
    if flash_fn is None:
        raise ImportError("sgl_kernel.flash_attn not available")

    batch_size, seq_len, num_heads, head_dim = query.shape

    q = query.reshape(-1, num_heads, head_dim)
    k = key.reshape(-1, num_heads, head_dim)
    v = value.reshape(-1, num_heads, head_dim)

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
    """SageAttention. Input: (batch, seq, heads, head_dim)"""
    sage_fn = _get_sglang_sage_attn()
    if sage_fn is None:
        raise ImportError("sageattention not available")

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
    """PyTorch SDPA. Input: (batch, seq, heads, head_dim)"""
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
# Backend Detection & Mapping
# =============================================================================

# User input → canonical backend type
_CANONICAL_MAP = {
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
    "sdpa": "native",
    "native": "native",
    "torch_sdpa": "native",
    "sglang_sdpa": "native",
}

# Canonical type → SGLang implementation
_SGLANG_IMPL = {
    "flash": (sglang_flash_attention, _get_sglang_flash_attn),
    "sage": (sglang_sage_attention, _get_sglang_sage_attn),
    "native": (sdpa_attention, lambda: True),
}


def _get_model_supported_backends(model) -> set[str]:
    """
    Get backends the model supports by probing set_attention_backend.
    Returns set of supported backend names.
    """
    if not hasattr(model, "set_attention_backend"):
        return set()

    # Try invalid backend to get list from error message
    try:
        model.set_attention_backend("__invalid__")
        return set()  # Shouldn't happen
    except Exception as e:
        error_msg = str(e)
        # Parse: "must be one of the following: flash, sage, native, ..."
        if "must be one of the following:" in error_msg:
            backends_str = error_msg.split("must be one of the following:")[-1]
            backends = {b.strip() for b in backends_str.split(",")}
            return backends
        return set()


def _check_backend_supported(supported: set[str], canonical: str) -> bool:
    """Check if canonical backend type is in model's supported list."""
    if canonical == "flash":
        return any(b in supported for b in ["flash", "_flash_3", "flash_varlen", "_flash_varlen_3"])
    elif canonical == "sage":
        return any(b in supported for b in ["sage", "sage_varlen", "_sage_qk_int8_pv_fp8_cuda"])
    elif canonical == "native":
        return "native" in supported
    return False


# =============================================================================
# Main API
# =============================================================================

def apply_sglang_attention(pipe: Any, backend: str = "auto") -> None:
    """
    Apply SGLang attention backend to diffusers pipeline.

    Flow:
    1. Query model's supported backends
    2. Map requested backend to canonical type (flash/sage/native)
    3. If model supports it AND SGLang has implementation → use SGLang
    4. Fall back to SDPA

    Args:
        pipe: Diffusers pipeline
        backend: "auto", "fa", "flash", "sage", "sdpa", etc.
    """
    # Find model
    model = None
    model_name = None
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        model = pipe.transformer
        model_name = "transformer"
    elif hasattr(pipe, "unet") and pipe.unet is not None:
        model = pipe.unet
        model_name = "unet"

    if model is None:
        logger.warning("No transformer/unet found in pipeline")
        return

    # Get model's supported backends
    supported = _get_model_supported_backends(model)
    if supported:
        logger.debug("Model supports backends: %s", supported)

    # Map to canonical type
    canonical = _CANONICAL_MAP.get(backend, backend)

    # Auto-select best available
    if backend == "auto" or canonical == "auto":
        for try_canonical in ["flash", "sage", "native"]:
            impl_fn, check_fn = _SGLANG_IMPL[try_canonical]
            if check_fn() is not None and (not supported or _check_backend_supported(supported, try_canonical)):
                canonical = try_canonical
                break
        else:
            canonical = "native"
        logger.info("Auto-selected: %s", canonical)

    # Check if model supports this backend
    if supported and not _check_backend_supported(supported, canonical):
        logger.warning(
            "Model doesn't support '%s', falling back to SDPA",
            canonical
        )
        canonical = "native"

    # Get SGLang implementation
    impl_fn, check_fn = _SGLANG_IMPL.get(canonical, (sdpa_attention, lambda: True))

    # Verify SGLang backend is available
    if check_fn() is None:
        logger.warning("SGLang %s not available, using SDPA", canonical)
        impl_fn = sdpa_attention
        canonical = "native"

    # Patch diffusers dispatch
    _patch_attention_dispatch(impl_fn, canonical)


def _patch_attention_dispatch(attn_fn, backend_name: str) -> None:
    """Patch diffusers' attention dispatch."""
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
        logger.info("Using SGLang %s attention", backend_name)

    except Exception as e:
        logger.warning("Could not patch attention dispatch: %s", e)
