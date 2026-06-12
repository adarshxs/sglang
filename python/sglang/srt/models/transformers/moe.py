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
"""Transformers modeling backend mixin for mixture-of-experts models."""

import logging
from collections.abc import Iterable
from typing import List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.transformers.utils import (
    _getattr_first,
    log_replacement,
    maybe_prefix,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import direct_register_custom_op

logger = logging.getLogger(__name__)

_TRANSFORMERS_MOE_LAYERS: dict[str, "TransformersFusedMoE"] = {}


class TransformersFusedMoE(nn.Module):
    """FusedMoE wrapper for the Transformers modeling backend.

    Wraps SGLang's native MoE implementation and exposes the
    ``(hidden_states, topk_ids, topk_weights)`` signature expected by
    Transformers' ``experts.forward()``.  A registered custom op
    (``torch.ops.sglang.transformers_moe_forward``) is used so that
    ``torch.compile`` can properly graph-break around the MoE kernel.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        reduce_results: bool,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        activation: str,
        with_bias: bool,
        expert_mapping: list,
    ) -> None:
        super().__init__()
        num_redundant = get_global_server_args().ep_num_redundant_experts
        experts_cls = get_moe_impl_class(quant_config)
        self.experts = experts_cls(
            num_experts=num_experts + num_redundant,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=reduce_results,
            quant_config=quant_config,
            activation=activation,
            with_bias=with_bias,
            prefix=prefix,
        )
        self.layer_name = prefix
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self._expert_mapping = expert_mapping
        _TRANSFORMERS_MOE_LAYERS[prefix] = self

    @property
    def tp_size(self) -> int:
        return getattr(self.experts, "moe_tp_size", 1)

    @property
    def ep_size(self) -> int:
        return getattr(self.experts, "moe_ep_size", 1)

    def maybe_all_reduce_tensor_model_parallel(
        self, output: torch.Tensor
    ) -> torch.Tensor:
        if self.tp_size > 1:
            return tensor_model_parallel_all_reduce(output)
        return output

    def get_expert_weights(self):
        return getattr(self.experts, "get_expert_weights", lambda: None)()

    def get_moe_weights(self) -> list[torch.Tensor]:
        num_local = getattr(self.experts, "num_local_experts", self.num_experts)
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ("correction_bias",)
            and filter_moe_weight_param_global_expert(name, x, num_local)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)
        if hidden_states.is_cuda:
            return torch.ops.sglang.transformers_moe_forward(
                hidden_states,
                topk_ids,
                topk_weights,
                self.layer_name,
            )
        return _transformers_moe_forward(
            hidden_states,
            topk_ids,
            topk_weights,
            self.layer_name,
        )

    # Checkpoints saved with Transformers >= v5 store all experts of a layer
    # as one 3D tensor in nn.Linear orientation: ``gate_up_proj`` is
    # [num_experts, 2 * intermediate_size, hidden_size] with the gate halves
    # first, and ``down_proj`` is [num_experts, hidden_size, intermediate_size].
    _FUSED_EXPERT_WEIGHTS = {
        "gate_up_proj": ("w1", "w3"),
        "down_proj": ("w2",),
    }

    def _load_fused_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        param_dict: dict[str, torch.nn.Parameter],
    ) -> bool:
        """Load one fused experts tensor by feeding per-expert slices through
        the regular FusedMoE weight loader. Returns False when *name* is not
        a fused experts weight."""
        base = name.removeprefix("experts.").removesuffix(".weight")
        shard_ids = self._FUSED_EXPERT_WEIGHTS.get(base)
        if shard_ids is None or loaded_weight.ndim != 3:
            return False

        param_name = (
            "experts.w13_weight" if base == "gate_up_proj" else "experts.w2_weight"
        )
        param = param_dict.get(param_name)
        if param is None:
            return False
        weight_loader = getattr(param, "weight_loader", default_weight_loader)

        # Normalize deviating checkpoints to nn.Linear orientation. The input
        # dimension is hidden_size for gate_up_proj and intermediate_size
        # (i.e. not hidden_size) for down_proj.
        in_dim_is_hidden = base == "gate_up_proj"
        if (loaded_weight.shape[-1] == self.hidden_size) != in_dim_is_hidden:
            loaded_weight = loaded_weight.transpose(-1, -2)

        # FusedMoE's loader dispatches on the weight name; use the canonical
        # per-expert spelling so the slices load as plain model weights.
        weight_name = f"{base}.weight"
        for expert_id in range(self.num_experts):
            expert_weight = loaded_weight[expert_id]
            shards = (
                expert_weight.chunk(len(shard_ids), dim=0)
                if len(shard_ids) > 1
                else (expert_weight,)
            )
            for shard_id, shard in zip(shard_ids, shards):
                weight_loader(
                    param, shard, weight_name, shard_id=shard_id, expert_id=expert_id
                )
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        param_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if self._load_fused_expert_weight(name, loaded_weight, param_dict):
                loaded.add(name)
                continue
            matched = False
            for param_name, weight_name, expert_id, shard_id in self._expert_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = param_dict.get(mapped_name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                except TypeError:
                    weight_loader(param, loaded_weight)
                loaded.add(name)
                matched = True
                break
            if not matched:
                direct_name = name if name in param_dict else f"experts.{name}"
                if direct_name in param_dict:
                    param = param_dict[direct_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    try:
                        weight_loader(param, loaded_weight)
                    except TypeError:
                        default_weight_loader(param, loaded_weight)
                    loaded.add(name)
                else:
                    logger.warning(
                        "MoE weight '%s' in layer '%s' could not be matched to any "
                        "parameter and will be skipped.",
                        name,
                        self.layer_name,
                    )
        return loaded


def _transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = _TRANSFORMERS_MOE_LAYERS[layer_name]
    # Record expert distribution for EPLB
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )

    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(self.experts.layer_id):
        recorder.on_select_experts(topk_ids=topk_ids)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=topk_weights,
    )
    return self.experts(hidden_states.clone(), topk_output)


def _transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="transformers_moe_forward",
    op_func=_transformers_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_transformers_moe_forward_fake,
)

try:
    from sglang.srt.compilation.compilation_config import SPLIT_OPS

    _MOE_SPLIT_OP = "sglang.transformers_moe_forward"
    if _MOE_SPLIT_OP not in SPLIT_OPS:
        SPLIT_OPS.append(_MOE_SPLIT_OP)
except ImportError:
    pass


class MoEMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_model_config_for_expert_location(
        cls, config
    ) -> Optional[ModelConfigForExpertLocation]:
        text_config = getattr(config, "text_config", config)
        num_experts = _getattr_first(
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        if num_experts is None:
            return None
        num_groups = getattr(text_config, "n_group", None)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=num_groups,
        )

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        return {
            fused.experts.layer_id: fused.get_moe_weights() for fused in self.moe_layers
        }

    def _get_expert_mapping(self, num_experts: int) -> List[Tuple[str, str, int, str]]:
        ckpt_names = [
            ("gate_proj", "down_proj", "up_proj"),
            ("w1", "w2", "w3"),
            ("linear", "linear_1", "linear_v"),
        ]
        mapping: list = []
        for gate, down, up in ckpt_names:
            mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate,
                    ckpt_down_proj_name=down,
                    ckpt_up_proj_name=up,
                    num_experts=num_experts,
                )
            )
        # AutoWeightsLoader dispatches to TransformersFusedMoE (which IS the
        # ``experts`` module) so the incoming weight names have the "experts."
        # prefix already stripped.  Remove it from weight_name in the mapping.
        mapping = [
            (pn, wn.removeprefix("experts."), eid, sid) for pn, wn, eid, sid in mapping
        ]
        return mapping

    def recursive_replace(self):
        """Replace experts modules with TransformersFusedMoE, then call
        super().recursive_replace() for Linear/RMSNorm replacement."""
        text_config = self.text_config

        num_experts = _getattr_first(
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        assert num_experts is not None, "Cannot determine num_experts from config."

        top_k = _getattr_first(text_config, ("num_experts_per_tok", "top_k"))
        assert top_k is not None, "Cannot determine top_k from config."

        hidden_size = text_config.hidden_size
        intermediate_size = _getattr_first(
            text_config,
            ("moe_intermediate_size", "intermediate_size"),
        )
        assert intermediate_size is not None, "Cannot determine intermediate_size."

        num_shared_experts = _getattr_first(
            text_config,
            ("n_shared_experts", "moe_num_shared_experts"),
            0,
        )
        reduce_results = num_shared_experts == 0

        renormalize = getattr(text_config, "norm_topk_prob", top_k > 1)

        # Activation function
        activation = "silu"
        wrapped_arch = self.config.architectures[0].lower()
        if "gptoss" in wrapped_arch:
            activation = "swigluoai"
        elif "grok1" in wrapped_arch:
            activation = "gelu"

        # Expert mapping for AutoWeightsLoader
        expert_mapping = self._get_expert_mapping(num_experts)

        # EPLB / EP tracking
        num_redundant = get_global_server_args().ep_num_redundant_experts
        ep_size = get_moe_expert_parallel_world_size()

        self.mlp_moe_layers: list[nn.Module] = []
        self.moe_layers: list[TransformersFusedMoE] = []
        self.num_moe_layers = 0
        self.num_logical_experts = num_experts
        self.num_physical_experts = num_experts + num_redundant
        self.num_local_physical_experts = self.num_physical_experts // max(ep_size, 1)
        self.num_shared_experts = num_shared_experts
        self.num_redundant_experts = num_redundant

        def _add_all_reduce(mlp: nn.Module):
            class MLPWithAllReduce(mlp.__class__):
                def forward(self, *args, **kwargs):
                    output = super().forward(*args, **kwargs)
                    return self.experts.maybe_all_reduce_tensor_model_parallel(output)

            mlp.__class__ = MLPWithAllReduce

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)

                is_modulelist = isinstance(child_module, nn.ModuleList)
                params = list(child_module.parameters())
                is_3d = len(params) > 0 and all(p.ndim == 3 for p in params)

                if child_name == "experts" and (is_modulelist or is_3d):
                    mlp = module
                    experts = child_module

                    has_bias = any("bias" in n for n, _ in experts.named_parameters())

                    nonlocal reduce_results
                    if reduce_results:
                        if any("shared_expert" in n for n, _ in mlp.named_parameters()):
                            reduce_results = False
                            self.num_shared_experts = 1

                    layer_id = self.num_moe_layers

                    fused_experts = TransformersFusedMoE(
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        layer_id=layer_id,
                        reduce_results=reduce_results,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                        activation=activation,
                        with_bias=has_bias,
                        expert_mapping=expert_mapping,
                    )
                    mlp.experts = fused_experts
                    log_replacement(qual_name, experts, fused_experts)

                    self.mlp_moe_layers.append(mlp)
                    self.moe_layers.append(fused_experts)
                    self.num_moe_layers += 1

                    if not reduce_results and (
                        fused_experts.tp_size > 1 or fused_experts.ep_size > 1
                    ):
                        _add_all_reduce(mlp)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")
        super().recursive_replace()
