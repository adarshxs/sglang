import ctypes
import os

import torch

if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

from sgl_kernel.version import __version__

if torch.version.cuda:
    from sgl_kernel.ops import (
        apply_rope_with_cos_sin_cache_inplace,
        bmm_fp8,
        build_tree_kernel,
        build_tree_kernel_efficient,
        cublas_grouped_gemm,
        custom_dispose,
        custom_reduce,
        fp8_blockwise_scaled_mm,
        fp8_scaled_mm,
        fused_add_rmsnorm,
        gelu_and_mul,
        gelu_tanh_and_mul,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        get_graph_buffer_ipc_meta,
        init_custom_reduce,
        int8_scaled_mm,
        lightning_attention_decode,
        min_p_sampling_from_probs,
        moe_align_block_size,
        register_graph_buffers,
        rmsnorm,
        sampling_scaling_penalties,
        sgl_per_token_group_quant_fp8,
        silu_and_mul,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

else:
    assert torch.version.hip

    from sgl_kernel.ops import (
        all_reduce_reg,
        all_reduce_unreg,
        allocate_meta_buffer,
        apply_rope_with_cos_sin_cache_inplace,
        bmm_fp8,
        dispose,
        fp8_scaled_mm,
        fused_add_rmsnorm,
        gelu_and_mul,
        gelu_tanh_and_mul,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        get_graph_buffer_ipc_meta,
        get_meta_buffer_ipc_handle,
        init_custom_ar,
        int8_scaled_mm,
        lightning_attention_decode,
        meta_size,
        min_p_sampling_from_probs,
        moe_align_block_size,
        register_buffer,
        register_graph_buffers,
        rmsnorm,
        sampling_scaling_penalties,
        silu_and_mul,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


__all__ = [
    "__version__",
    "apply_rope_with_cos_sin_cache_inplace",
    "bmm_fp8",
    "cublas_grouped_gemm",
    "custom_dispose",
    "custom_reduce",
    "build_tree_kernel_efficient",
    "build_tree_kernel",
    "fp8_blockwise_scaled_mm",
    "fp8_scaled_mm",
    "fused_add_rmsnorm",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "get_graph_buffer_ipc_meta",
    "init_custom_reduce",
    "int8_scaled_mm",
    "lightning_attention_decode",
    "min_p_sampling_from_probs",
    "moe_align_block_size",
    "register_graph_buffers",
    "rmsnorm",
    "sampling_scaling_penalties",
    "sgl_per_token_group_quant_fp8",
    "silu_and_mul",
    "top_k_renorm_prob",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_prob",
    "tree_speculative_sampling_target_only",
]
