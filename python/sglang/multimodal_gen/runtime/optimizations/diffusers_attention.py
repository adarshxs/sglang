# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention processors for diffusers pipelines.

This module provides attention processors that use SGLang's optimized attention backends
(FlashAttention, SageAttention, SDPA) with diffusers' set_attn_processor() API.
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Available attention backends that work well with diffusers
_DIFFUSERS_COMPATIBLE_BACKENDS = {
    AttentionBackendEnum.FA,
    AttentionBackendEnum.SAGE_ATTN,
    AttentionBackendEnum.SAGE_ATTN_THREE,
    AttentionBackendEnum.TORCH_SDPA,
}

# Map CLI backend names to internal names
_BACKEND_NAME_MAP = {
    # CLI names -> internal names
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
    """Get list of available attention backends for diffusers.

    Returns:
        List of backend names that are available and compatible with diffusers.
    """
    available = []

    # Check FlashAttention
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func  # noqa: F401

        available.append("flash_attn")
    except ImportError:
        pass

    # Check SageAttention
    try:
        from sageattention import sageattn  # noqa: F401

        available.append("sage_attn")
    except ImportError:
        pass

    # Check SageAttention 3
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda  # noqa: F401

        available.append("sage_attn_3")
    except ImportError:
        pass

    # SDPA is always available with PyTorch >= 2.0
    if hasattr(F, "scaled_dot_product_attention"):
        available.append("sdpa")

    return available


class SGLangAttnProcessor:
    """
    Diffusers-compatible attention processor using SGLang backends.

    This processor replaces diffusers' default attention with SGLang's optimized
    implementations (FlashAttention, SageAttention, or SDPA).

    Usage:
        pipe.unet.set_attn_processor(SGLangAttnProcessor(backend="fa"))
    """

    def __init__(
        self,
        backend: str = "auto",
        softmax_scale: float | None = None,
    ):
        """
        Initialize the SGLang attention processor.

        Args:
            backend: Attention backend to use. Options:
                CLI names (from --attention-backend):
                - "fa", "fa3", "fa4": FlashAttention
                - "torch_sdpa": PyTorch SDPA
                - "sage_attn": SageAttention
                - "sage_attn_three": SageAttention 3 (FP8)
                - "auto": Automatically select best available
                Internal names also accepted:
                - "flash_attn", "sdpa", "sage_attn_3"
            softmax_scale: Optional scale for softmax. If None, uses 1/sqrt(head_dim).
        """
        # Normalize backend name (CLI names -> internal names)
        self.backend_name = _BACKEND_NAME_MAP.get(backend, backend)
        self.softmax_scale = softmax_scale
        self._attn_fn: Callable | None = None
        self._backend_initialized = False

    def _init_backend(self, head_dim: int) -> None:
        """Lazily initialize the attention backend."""
        if self._backend_initialized:
            return

        backend = self.backend_name
        if backend == "auto":
            available = get_available_attention_backends()
            if "flash_attn" in available:
                backend = "flash_attn"
            elif "sage_attn" in available:
                backend = "sage_attn"
            else:
                backend = "sdpa"
            logger.info("Auto-selected attention backend: %s", backend)

        if backend == "flash_attn":
            self._attn_fn = self._flash_attn_forward
            logger.debug("Using FlashAttention backend")
        elif backend == "sage_attn":
            self._attn_fn = self._sage_attn_forward
            logger.debug("Using SageAttention backend")
        elif backend == "sage_attn_3":
            self._attn_fn = self._sage_attn_3_forward
            logger.debug("Using SageAttention 3 backend")
        else:
            self._attn_fn = self._sdpa_forward
            logger.debug("Using SDPA backend")

        if self.softmax_scale is None:
            self.softmax_scale = head_dim**-0.5

        self._backend_initialized = True

    def _flash_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using FlashAttention."""
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        batch_size, seq_len, num_heads, head_dim = query.shape

        # FlashAttention expects (total_tokens, num_heads, head_dim)
        q = query.reshape(-1, num_heads, head_dim)
        k = key.reshape(-1, num_heads, head_dim)
        v = value.reshape(-1, num_heads, head_dim)

        # Create cumulative sequence lengths
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=query.device,
        )

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            seq_len,
            seq_len,
            softmax_scale=self.softmax_scale,
            causal=False,
        )

        return output.reshape(batch_size, seq_len, num_heads, head_dim)

    def _sage_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using SageAttention."""
        from sageattention import sageattn

        # SageAttention expects (batch, heads, seq_len, head_dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        output = sageattn(q, k, v, is_causal=False, smooth_k=True)

        return output.transpose(1, 2)

    def _sage_attn_3_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using SageAttention 3 (FP8)."""
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        # SageAttention expects (batch, heads, seq_len, head_dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        output = sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=False, smooth_k=True)

        return output.transpose(1, 2)

    def _sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using PyTorch SDPA."""
        # SDPA expects (batch, heads, seq_len, head_dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        )

        return output.transpose(1, 2)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention using SGLang backend.

        This follows the diffusers AttentionProcessor interface.
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Initialize backend with actual head_dim
        self._init_backend(head_dim)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply SGLang attention backend
        hidden_states = self._attn_fn(query, key, value)

        # Reshape back
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SGLangAttnProcessor2_0(SGLangAttnProcessor):
    """
    SGLang attention processor for Attention 2.0 (used in SD 2.x, SDXL, etc.).

    This handles the slightly different attention interface used in newer models.
    """

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Process attention for Attention 2.0 models."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Initialize backend with actual head_dim
        self._init_backend(head_dim)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply SGLang attention backend
        hidden_states = self._attn_fn(query, key, value)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_sglang_attention(
    pipe: Any,
    backend: str = "auto",
) -> None:
    """
    Apply SGLang attention processors to a diffusers pipeline.

    This replaces the attention implementations in the pipeline's transformer/unet
    with SGLang's optimized backends.

    Args:
        pipe: A diffusers pipeline (e.g., StableDiffusionPipeline, FluxPipeline)
        backend: Attention backend to use ("auto", "flash_attn", "sage_attn", "sdpa")

    Example:
        ```python
        from diffusers import StableDiffusionPipeline
        from sglang.multimodal_gen.runtime.optimizations import apply_sglang_attention

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        apply_sglang_attention(pipe, backend="flash_attn")
        ```
    """
    processor = SGLangAttnProcessor2_0(backend=backend)
    models_to_patch = []

    # Check for transformer (DiT models like FLUX, SD3)
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models_to_patch.append(("transformer", pipe.transformer))

    # Check for unet (UNet models like SD 1.5, SDXL)
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models_to_patch.append(("unet", pipe.unet))

    if not models_to_patch:
        logger.warning(
            "No transformer or unet found in pipeline, cannot apply attention processors"
        )
        return

    for name, model in models_to_patch:
        if hasattr(model, "set_attn_processor"):
            model.set_attn_processor(processor)
            logger.info(
                "Applied SGLang attention processor (%s backend) to %s",
                backend,
                name,
            )
        else:
            logger.warning("%s does not support set_attn_processor", name)

