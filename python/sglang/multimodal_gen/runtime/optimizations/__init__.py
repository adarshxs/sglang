# SPDX-License-Identifier: Apache-2.0
"""
SGLang optimizations for diffusers backend.

Provides attention backend integration and data parallelism.
"""

from sglang.multimodal_gen.runtime.optimizations.data_parallel import (
    DataParallelWrapper,
    apply_data_parallel,
    enable_device_map_parallel,
    get_available_gpus,
)
from sglang.multimodal_gen.runtime.optimizations.diffusers_attention import (
    apply_sglang_attention,
    sdpa_attention,
    sglang_flash_attention,
    sglang_sage_attention,
)

__all__ = [
    # Attention
    "apply_sglang_attention",
    "sglang_flash_attention",
    "sglang_sage_attention",
    "sdpa_attention",
    # Data parallelism
    "DataParallelWrapper",
    "apply_data_parallel",
    "enable_device_map_parallel",
    "get_available_gpus",
]
