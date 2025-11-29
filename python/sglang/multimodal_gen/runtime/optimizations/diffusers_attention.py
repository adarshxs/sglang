# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention processors for diffusers pipelines.

This module provides a simple, robust attention processor that uses SGLang's
optimized backends (FlashAttention, SageAttention, SDPA) where possible,
falling back gracefully to diffusers' defaults on any error.
"""

from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Map CLI backend names to internal names
_BACKEND_NAME_MAP = {
    "fa": "flash_attn",
    "fa3": "flash_attn",
    "fa4": "flash_attn",
    "flash_attn": "flash_attn",
    "torch_sdpa": "sdpa",
    "sdpa": "sdpa",
    "sage_attn": "sage_attn",
    "sage_attn_three": "sage_attn_3",
    "sage_attn_3": "sage_attn_3",
    "auto": "auto",
}


def get_available_attention_backends() -> list[str]:
    """Get list of available attention backends."""
    available = []

    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func  # noqa: F401

        available.append("flash_attn")
    except ImportError:
        pass

    try:
        from sageattention import sageattn  # noqa: F401

        available.append("sage_attn")
    except ImportError:
        pass

    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda  # noqa: F401

        available.append("sage_attn_3")
    except ImportError:
        pass

    if hasattr(F, "scaled_dot_product_attention"):
        available.append("sdpa")

    return available


def _get_attn_func(backend: str):
    """Get the attention function for the specified backend.

    Returns None if the backend is not available.
    """
    backend = _BACKEND_NAME_MAP.get(backend, backend)

    if backend == "auto":
        available = get_available_attention_backends()
        if "flash_attn" in available:
            backend = "flash_attn"
        elif "sage_attn" in available:
            backend = "sage_attn"
        else:
            backend = "sdpa"

    if backend == "flash_attn":
        try:
            from sgl_kernel.flash_attn import flash_attn_varlen_func

            return ("flash_attn", flash_attn_varlen_func)
        except ImportError:
            return None

    if backend == "sage_attn":
        try:
            from sageattention import sageattn

            return ("sage_attn", sageattn)
        except ImportError:
            return None

    if backend == "sage_attn_3":
        try:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda

            return ("sage_attn_3", sageattn_qk_int8_pv_fp8_cuda)
        except ImportError:
            return None

    if backend == "sdpa":
        return ("sdpa", F.scaled_dot_product_attention)

    return None


def apply_sglang_attention(
    pipe: Any,
    backend: str = "auto",
) -> None:
    """
    Try to apply SGLang attention to a diffusers pipeline.

    This is a best-effort optimization. If anything fails, the pipeline
    continues using its default attention implementation.

    Args:
        pipe: A diffusers pipeline
        backend: Attention backend ("auto", "fa", "torch_sdpa", "sage_attn", etc.)
    """
    # Get the attention function
    attn_info = _get_attn_func(backend)
    if attn_info is None:
        logger.warning(
            "Requested attention backend '%s' not available, skipping optimization",
            backend,
        )
        return

    backend_name, attn_func = attn_info

    # Find models to patch
    models = []
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models.append(("transformer", pipe.transformer))
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models.append(("unet", pipe.unet))

    if not models:
        logger.debug("No transformer or unet found in pipeline")
        return

    for name, model in models:
        try:
            if not hasattr(model, "set_attn_processor"):
                logger.info("%s does not support set_attn_processor, skipping", name)
                continue

            # Create and set processor
            processor = SGLangAttnProcessor(backend_name, attn_func)
            model.set_attn_processor(processor)
            logger.info("Applied SGLang attention (%s) to %s", backend_name, name)

        except Exception as e:
            logger.info(
                "Could not apply SGLang attention to %s (%s), using default", name, e
            )


class SGLangAttnProcessor:
    """
    Minimal attention processor that wraps SGLang attention backends.

    Designed to be robust - if anything goes wrong during forward pass,
    it falls back to standard SDPA.
    """

    def __init__(self, backend_name: str, attn_func: Any):
        self.backend_name = backend_name
        self.attn_func = attn_func
        self._warned = False

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention. Falls back to SDPA on any error.
        """
        try:
            return self._forward(attn, hidden_states, *args, **kwargs)
        except Exception as e:
            if not self._warned:
                logger.warning(
                    "SGLang attention failed (%s), falling back to SDPA. "
                    "This warning will only show once.",
                    e,
                )
                self._warned = True
            return self._fallback_sdpa(attn, hidden_states, *args, **kwargs)

    def _forward(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Core attention computation with SGLang backend."""
        batch_size = hidden_states.shape[0]
        residual = hidden_states

        # Handle spatial norm if present
        temb = kwargs.get("temb")
        if hasattr(attn, "spatial_norm") and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Group norm
        if hasattr(attn, "group_norm") and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # QKV projections
        query = attn.to_q(hidden_states)

        kv_input = hidden_states
        if encoder_hidden_states is not None:
            kv_input = encoder_hidden_states
            if hasattr(attn, "norm_cross") and attn.norm_cross:
                kv_input = attn.norm_encoder_hidden_states(kv_input)

        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        # Reshape to (batch, seq, heads, head_dim)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Run attention based on backend
        hidden_states = self._run_attention(query, key, value, head_dim)

        # Reshape back
        hidden_states = hidden_states.reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Handle 4D input
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Residual connection
        if hasattr(attn, "residual_connection") and attn.residual_connection:
            hidden_states = hidden_states + residual

        if hasattr(attn, "rescale_output_factor"):
            hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _run_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        """Run attention with the configured backend."""
        batch_size, seq_len = query.shape[:2]
        softmax_scale = head_dim**-0.5

        if self.backend_name == "flash_attn":
            # FlashAttention: (batch*seq, heads, head_dim)
            q = query.reshape(-1, query.shape[2], query.shape[3])
            k = key.reshape(-1, key.shape[2], key.shape[3])
            v = value.reshape(-1, value.shape[2], value.shape[3])

            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=query.device
            )

            out = self.attn_func(
                q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len,
                softmax_scale=softmax_scale, causal=False
            )
            return out.reshape(batch_size, seq_len, -1, head_dim)

        elif self.backend_name in ("sage_attn", "sage_attn_3"):
            # SageAttention: (batch, heads, seq, head_dim)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = self.attn_func(q, k, v, is_causal=False, smooth_k=True)
            return out.transpose(1, 2)

        else:  # sdpa
            # SDPA: (batch, heads, seq, head_dim)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=softmax_scale
            )
            return out.transpose(1, 2)

    def _fallback_sdpa(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fallback to standard SDPA when SGLang backend fails."""
        batch_size = hidden_states.shape[0]
        residual = hidden_states

        temb = kwargs.get("temb")
        if hasattr(attn, "spatial_norm") and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if hasattr(attn, "group_norm") and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Use SDPA
        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if hasattr(attn, "residual_connection") and attn.residual_connection:
            hidden_states = hidden_states + residual

        if hasattr(attn, "rescale_output_factor"):
            hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# Aliases for compatibility
SGLangAttnProcessor2_0 = SGLangAttnProcessor
