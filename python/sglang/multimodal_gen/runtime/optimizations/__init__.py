# SPDX-License-Identifier: Apache-2.0
"""
SGLang optimizations for diffusers backend.

This module provides optimization utilities that can be applied to diffusers pipelines
to leverage SGLang's high-performance backends.

Available optimizations:
- Attention backends: FlashAttention, SageAttention, SDPA
- Data parallelism: Multi-GPU batch processing
- Memory optimizations: VAE tiling/slicing, CPU offloading
- Compilation: torch.compile for transformer/unet
"""

from sglang.multimodal_gen.runtime.optimizations.data_parallel import (
    DataParallelWrapper,
    apply_data_parallel,
    enable_device_map_parallel,
    get_available_gpus,
)
from sglang.multimodal_gen.runtime.optimizations.diffusers_attention import (
    SGLangAttnProcessor,
    SGLangAttnProcessor2_0,
    apply_sglang_attention,
    get_available_attention_backends,
)

__all__ = [
    # Attention processors
    "SGLangAttnProcessor",
    "SGLangAttnProcessor2_0",
    "apply_sglang_attention",
    "get_available_attention_backends",
    # Data parallelism
    "DataParallelWrapper",
    "apply_data_parallel",
    "enable_device_map_parallel",
    "get_available_gpus",
]

