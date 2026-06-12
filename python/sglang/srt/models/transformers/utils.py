# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers
"""Transformers modeling backend utilities."""

import logging
from contextlib import contextmanager
from typing import Literal, Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


def can_enable_torch_compile(config: PretrainedConfig) -> bool:
    """Check whether the model config is compatible with torch.compile.

    Dynamic rope scaling triggers data-dependent control flow that prevents
    capturing a single computation graph, so we disable compilation for it.
    """
    text_config = getattr(config, "text_config", config)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "dynamic":
            return False
    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        if isinstance(next(iter(rope_params.values()), None), dict):
            return not any(
                rp.get("rope_type") == "dynamic" for rp in rope_params.values()
            )
        if rope_params.get("rope_type") == "dynamic":
            return False
    return True


def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def _getattr_first(obj, names, default=None):
    """Return the first existing attribute from *names*, else *default*."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


@contextmanager
def _init_on_device_without_buffers(device: torch.device):
    """Initialize model parameters on *device* while leaving buffers on CPU.
    Adapted from ``accelerate``."""
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]


def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: Optional[QuantizationConfig] = None,
    *,
    prefix: str = "",
) -> Union[ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]:
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    sglang_linear_cls, linear_kwargs = {
        "colwise": (ColumnParallelLinear, {}),
        "colwise_rep": (ColumnParallelLinear, {"gather_output": True}),
        "rowwise": (RowParallelLinear, {}),
        "rowwise_rep": (RowParallelLinear, {"input_is_parallel": False}),
        "replicate": (ReplicatedLinear, {}),
    }.get(style, (ReplicatedLinear, {}))

    class HFCompatibleLinear(sglang_linear_cls):
        @property
        def parent_cls(self) -> type:
            return sglang_linear_cls

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)[0]

    return HFCompatibleLinear(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        prefix=prefix,
        **linear_kwargs,
    )


# Styles that map onto SGLang's tensor-parallel linear layers.
# ``colwise_gather_output`` / ``rowwise_split_input`` are the Transformers v5
# spellings of ``colwise_rep`` / ``rowwise_rep``: the output is all-gathered
# (resp. the replicated input is split) at the layer boundary.
_TP_STYLE_ALIASES = {
    # Transformers v5 vocabulary
    "colwise_gather_output": "colwise_rep",
    "rowwise_split_input": "rowwise_rep",
    "packed_colwise": "colwise",
    "packed_rowwise": "rowwise",
    # Transformers v4 vocabulary
    "colwiseparallel": "colwise",
    "local_colwise": "colwise",
    "rowwiseparallel": "rowwise",
    "local_rowwise": "rowwise",
    "local_packed_rowwise": "rowwise",
    "isolated": "replicate",
    "local": "replicate",
    "replicated_with_grad_allreduce": "replicate",
}

# Styles targeting modules the backend parallelizes (or replicates) through
# other means: vocab embeddings are replaced by ``VocabParallelEmbedding``,
# MoE experts/routers by ``TransformersFusedMoE``, and norms stay replicated.
# These entries never apply to a plain ``nn.Linear``, so they are dropped
# from the plan instead of being treated as unknown styles.
_TP_STYLES_HANDLED_ELSEWHERE = frozenset(
    {
        "all_reduce",
        "embedding_colwise",
        "embedding_rowwise",
        "ep_router",
        "grouped_gemm",
        "megamoe_experts",
        "megamoe_router",
        "mla_kv_a_proj",
        "moe_identity_expert",
        "moe_tp_experts",
        "sequence_parallel",
    }
)


def _normalize_tp_style(style: str) -> Optional[Style]:
    """Map a Transformers TP style onto SGLang's linear styles.

    Returns None for styles whose modules the backend handles through other
    means (see ``_TP_STYLES_HANDLED_ELSEWHERE``); the caller drops them.
    Unknown styles fall back to "replicate" when running on a single rank,
    where every style degenerates to a plain linear, and raise otherwise so
    that tensor parallelism never silently produces wrong results.
    """
    style = style.lower().replace("-", "_")
    if style in _TP_STYLES_HANDLED_ELSEWHERE:
        return None
    style = _TP_STYLE_ALIASES.get(style, style)
    if style not in {"colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"}:
        if get_tensor_model_parallel_world_size() == 1:
            logger.warning(
                "Unknown TP style '%s' in the model's tensor-parallel plan; "
                "treating it as 'replicate' since tp_size is 1.",
                style,
            )
            return "replicate"
        raise ValueError(f"Unsupported TP style '{style}' for Transformers backend.")
    return style


def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> nn.Module:
    eps = _getattr_first(rms_norm, ("eps", "variance_epsilon"), 1e-6)
    kwargs = {"hidden_size": hidden_size, "eps": eps}
    weight_meta = getattr(rms_norm, "weight", None)
    if weight_meta is not None:
        kwargs["hidden_size"] = weight_meta.size(0)

    try:
        with torch.device("cpu"):
            weight_test = getattr(rms_norm.__class__(1), "weight", None)
    except Exception:
        weight_test = None
    is_gemma = weight_test is not None and torch.all(weight_test == 0)

    if is_gemma:
        base_cls = GemmaRMSNorm
        norm = base_cls(
            **{k: v for k, v in kwargs.items() if k in ("hidden_size", "eps")}
        )
    else:
        kwargs["has_weight"] = getattr(rms_norm, "with_scale", True)
        if weight_meta is not None:
            kwargs["weight_dtype"] = weight_meta.dtype
        else:
            kwargs["has_weight"] = False
        kwargs["cast_x_before_out_mul"] = (
            True  # match HF fp16-weight-multiply semantics
        )
        base_cls = RMSNorm
        norm = base_cls(**kwargs)

    # Wrap to handle 3D inputs from Transformers backbone (batch dim)
    class HFCompatibleRMSNorm(norm.__class__):
        def forward(self, x, *args, **kwargs):
            orig_shape = x.shape
            if x.ndim > 2:
                x = x.reshape(-1, x.shape[-1]).contiguous()
            result = super().forward(x, *args, **kwargs)
            if isinstance(result, tuple):
                return tuple(
                    (
                        r.reshape(orig_shape)
                        if torch.is_tensor(r) and r.shape != orig_shape
                        else r
                    )
                    for r in result
                )
            if torch.is_tensor(result) and result.shape != orig_shape:
                return result.reshape(orig_shape)
            return result

    norm.__class__ = HFCompatibleRMSNorm
    return norm
