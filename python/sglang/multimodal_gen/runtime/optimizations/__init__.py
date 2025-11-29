# SPDX-License-Identifier: Apache-2.0
"""SGLang optimizations for diffusers backend."""

from sglang.multimodal_gen.runtime.optimizations.diffusers_attention import (
    apply_sglang_attention,
    get_current_backend,
    reset,
)
from sglang.multimodal_gen.runtime.optimizations.data_parallel import (
    DataParallelWrapper,
    apply_data_parallel,
    enable_device_map_parallel,
    get_available_gpus,
)

__all__ = [
    # Attention
    "apply_sglang_attention",
    "get_current_backend",
    "reset",
    # Data parallel
    "DataParallelWrapper",
    "apply_data_parallel",
    "enable_device_map_parallel",
    "get_available_gpus",
]
