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
"""Transformers modeling backend base class."""

import inspect
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Optional, Union

import torch
import transformers
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.models.transformers.utils import (
    Style,
    _getattr_first,
    _init_on_device_without_buffers,
    _normalize_tp_style,
    can_enable_torch_compile,
    log_replacement,
    maybe_prefix,
    replace_linear_class,
    replace_rms_norm_class,
)
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper
from sglang.srt.utils import get_device
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config

logger = logging.getLogger(__name__)


def _resolve_attention_backend_model_cls(config: PretrainedConfig):
    """Resolve the class that ``AutoModel.from_config(config,
    trust_remote_code=True)`` will instantiate.

    Models shipping their own modeling code (``auto_map``) take precedence
    over a same-named class in the transformers library, mirroring
    ``AutoModel``'s resolution order.  This matters for the
    ``_supports_attention_backend`` check: custom modeling code usually
    hard-codes its attention classes and never consults
    ``ALL_ATTENTION_FUNCTIONS``, so checking the library class instead would
    let the wrapped model crash later with an opaque ``KeyError: 'sglang'``.
    """
    auto_map = getattr(config, "auto_map", {}) or {}
    for key in ("AutoModel", "AutoModelForCausalLM"):
        if key not in auto_map:
            continue
        try:
            return get_class_from_dynamic_module(
                auto_map[key],
                getattr(config, "_name_or_path", ""),
            )
        except Exception as e:
            logger.warning(
                "Failed to load dynamic module from auto_map[%s]: %s.",
                key,
                e,
            )
    return getattr(transformers, getattr(config, "architectures", [""])[0], None)


def sglang_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float = None,
    attention_instances: Optional[Mapping[int, RadixAttention]] = None,
    forward_batch: Optional[ForwardBatch] = None,
    **kwargs,
):
    self_attn: RadixAttention = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.scaling = float(scaling)

    # TP plans that all-gather the attention projections
    # (colwise_gather_output, e.g. OLMoE's q/k norms over all heads or
    # Phi3's fused qkv slicing) hand every rank the full head set. Slice
    # this rank's heads so attention and the KV cache stay head-sharded,
    # and gather the context afterwards for the rowwise_split_input o_proj.
    tp_size = get_tensor_model_parallel_world_size()
    heads_gathered = tp_size > 1 and query.shape[1] == self_attn.tp_q_head_num * tp_size
    if heads_gathered:
        tp_rank = get_tensor_model_parallel_rank()
        q_heads, kv_heads = self_attn.tp_q_head_num, self_attn.tp_k_head_num
        query = query[:, tp_rank * q_heads : (tp_rank + 1) * q_heads]
        if key.shape[1] == kv_heads * tp_size:
            key = key[:, tp_rank * kv_heads : (tp_rank + 1) * kv_heads]
            value = value[:, tp_rank * kv_heads : (tp_rank + 1) * kv_heads]

    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    context = self_attn.forward(query, key, value, forward_batch=forward_batch)
    if heads_gathered:
        context = tensor_model_parallel_all_gather(context)
    return context, None


ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward


_BASE_DYNAMIC_ARG_DIMS: dict[str, int] = {
    "input_ids": 0,
    "positions": 0,
    "input_embeds": 0,
}


class TransformersBase(nn.Module):
    torch_compile_dynamic_arg_dims: dict[str, int] = _BASE_DYNAMIC_ARG_DIMS

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",
            "model.transformer.": "model.",
            "model.model.": "model.",
            "model.lm_head.": "lm_head.",
            "model.score.": "classifier.",
            "model.classifier.": "classifier.",
            "transformer.": "model.",
            "model.": "model.",
            "lm_head.": "lm_head.",
            "score.": "classifier.",
            "classifier.": "classifier.",
            "": "model.",
        }
    )

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        mapper = WeightsMapper()
        for base in cls.__mro__:
            base_mapper = getattr(base, "hf_to_sglang_mapper", None)
            if base_mapper is not None:
                mapper = mapper | base_mapper
        cls.hf_to_sglang_mapper = mapper

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        self.quant_config = quant_config
        self.config = config
        self.text_config = get_hf_text_config(config)
        self.weight_mapper = self.hf_to_sglang_mapper
        self.pp_group = get_pp_group()

        # Weight loading attrs
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []
        self.skip_substrs.extend([".attn.bias", ".attn.masked_bias", ".masked_bias"])
        self.ignore_unexpected_prefixes.extend(["classifier.", "score."])

        if self.quant_config is not None:
            quant_method_name = self.quant_config.get_name()
            if "gptq" in quant_method_name:
                self.ignore_unexpected_suffixes.append(".bias")
            if "fp8" in quant_method_name:
                fp8_suffix_map = {".activation_scale": ".input_scale"}
                use_mxfp8 = bool(getattr(self.quant_config, "use_mxfp8", False))
                weight_block_size = getattr(
                    self.quant_config, "weight_block_size", None
                )
                if not use_mxfp8 and weight_block_size is None:
                    fp8_suffix_map[".weight_scale_inv"] = ".weight_scale"
                self.weight_mapper = self.weight_mapper | WeightsMapper(
                    orig_to_new_suffix=fp8_suffix_map
                )

        # Resolve model class for _supports_attention_backend check
        model_cls = _resolve_attention_backend_model_cls(config)

        supports_backend = (
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )

        # Initialize on meta device to avoid premature GPU allocation
        self.text_config._attn_implementation = "sglang"
        # Transformers creates modules in the config dtype (the torch_dtype
        # kwarg is deprecated and not honored by every version), so pin the
        # config and its sub-configs (text/vision towers) to SGLang's
        # requested dtype. Otherwise checkpoints whose config declares a
        # different dtype get mixed-dtype modules, which the fused kernels
        # silently miscompute.
        self.config.dtype = torch.get_default_dtype()
        for sub_config_name in getattr(self.config, "sub_configs", None) or {}:
            sub_config = getattr(self.config, sub_config_name, None)
            if sub_config is not None and sub_config.dtype != self.config.dtype:
                sub_config.dtype = self.config.dtype
        if supports_backend:
            with _init_on_device_without_buffers(torch.device("meta")):
                self.model: PreTrainedModel = AutoModel.from_config(
                    self.config,
                    torch_dtype=torch.get_default_dtype(),
                    trust_remote_code=True,
                )
        else:
            is_remote_code = (model_cls.__module__ or "").startswith(
                "transformers_modules"
            )
            hint = (
                "The checkpoint ships its own modeling code "
                f"({model_cls.__module__}) that hard-codes its attention "
                "classes instead of using transformers' attention interface, "
                "so SGLang cannot inject its attention backend. Use a "
                "checkpoint whose architecture is part of the transformers "
                "library, or SGLang's native implementation "
                "(--model-impl auto) if one exists."
                if is_remote_code
                else "Ask the model authors to adopt transformers' attention "
                "interface (ALL_ATTENTION_FUNCTIONS) to enable backend "
                "support."
            )
            raise ValueError(
                f"{model_cls.__name__} does not support custom attention "
                f"backends (_supports_attention_backend=False), which the "
                f"Transformers backend requires. {hint}"
            )

        # Checkpoints nest the backbone weights under the model's
        # base_model_prefix (e.g. "bert.embeddings.*" for BERT-family
        # classifiers); map such prefixes onto the wrapper's "model.".
        base_model_prefix = getattr(self.model, "base_model_prefix", "")
        if base_model_prefix and base_model_prefix != "model":
            self.weight_mapper = self.weight_mapper | WeightsMapper(
                orig_to_new_prefix={f"{base_model_prefix}.": "model."}
            )

        # Encoder-only models (e.g. BERT) flag their attention modules with
        # is_causal=False. Multimodal towers do too, so only consider
        # text-only configs.
        self.is_encoder_only = self.config is self.text_config and any(
            not getattr(m, "is_causal", True)
            for m in self.model.modules()
            if hasattr(m, "is_causal")
        )

        forward_params = inspect.signature(self.model.forward).parameters
        self._accepts_token_type_ids = "token_type_ids" in forward_params

        # RoBERTa-family embeddings offset position ids by padding_idx + 1,
        # but only when computing the ids themselves; replicate the offset
        # since the backend always passes explicit positions.
        self._position_offset = 0
        for module in self.model.modules():
            if hasattr(module, "create_position_ids_from_input_ids"):
                padding_idx = getattr(module, "padding_idx", None)
                if padding_idx is not None:
                    self._position_offset = padding_idx + 1
                break

        self.vocab_size = getattr(
            self.text_config,
            "vocab_size",
            self.model.get_input_embeddings().num_embeddings,
        )
        self.unpadded_vocab_size = self.vocab_size

        # Whether the lm_head is tied is governed by the top-level config:
        # nested sub-configs (e.g. the text_config of VL models) default
        # tie_word_embeddings to True even when the checkpoint has a real
        # lm_head and declares tie_word_embeddings=False at the top level.
        self.tie_word_embeddings = getattr(
            self.config,
            "tie_word_embeddings",
            getattr(self.text_config, "tie_word_embeddings", False),
        )

        # Embedding scale (e.g. Whisper)
        input_embeddings = self.model.get_input_embeddings()
        self.embed_scale = getattr(input_embeddings, "embed_scale", None)

        self.start_layer = 0
        self.end_layer = getattr(self.text_config, "num_hidden_layers", 0)

        # Pipeline parallel
        self.pipeline_parallel()
        # Module replacement (Linear → TP, RMSNorm → fused, MoE overridden by MoEMixin)
        tp_size = get_tensor_model_parallel_world_size()
        self.recursive_replace()
        # Attention instances
        self.attention_instances = self._create_attention_instances(tp_size)
        # Vocab embeddings
        self.replace_vocab_embed_class(self.model)

        # Initialize remaining meta-device parameters to real device tensors
        self._init_parameters(self.model)

        self.lm_head: Optional[ParallelLMHead] = None
        self.logits_processor: Optional[LogitsProcessor] = None
        self.pooler: Optional[Pooler] = None

        # EAGLE3 aux hidden states (see CausalMixin.set_eagle3_layers_to_capture)
        self.capture_aux_hidden_states = False
        self._aux_hidden_states: dict[int, torch.Tensor] = {}
        self._layers_to_capture: list[int] = []

        self._compile_compatible = can_enable_torch_compile(config)

    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""
        return self._compile_compatible

    def _get_decoder_layer_list(self) -> nn.ModuleList:
        """Return the text decoder's layer ModuleList (the one whose length
        matches num_hidden_layers; vision towers have their own lists)."""
        num_layers = self.text_config.num_hidden_layers
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList) and len(module) == num_layers:
                return module
        raise ValueError(
            f"Could not find a decoder layer list of length {num_layers} in "
            f"{type(self.model).__name__}."
        )

    def _install_aux_hidden_state_hooks(self, layers_to_capture: list[int]):
        """Capture the input hidden states of the given decoder layers
        (e.g. for EAGLE3 draft models) via forward pre-hooks."""
        layers = self._get_decoder_layer_list()

        def _make_hook(layer_idx: int):
            def _hook(module, args, kwargs):
                hidden_states = args[0] if args else kwargs["hidden_states"]
                # Drop the batch dimension to match SGLang's flat layout.
                self._aux_hidden_states[layer_idx] = hidden_states[0, ...]

            return _hook

        for layer_idx in layers_to_capture:
            if layer_idx in self._layers_to_capture:
                continue
            layers[layer_idx].register_forward_pre_hook(
                _make_hook(layer_idx), with_kwargs=True
            )
            self._layers_to_capture.append(layer_idx)
        self._layers_to_capture.sort()

    def _init_parameters(self, module: nn.Module):
        """Materialize any parameters still on the meta device."""
        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                new_param = nn.Parameter(
                    torch.empty_like(
                        param.data,
                        device=get_device(),
                    )
                )
                setattr(module, name, new_param)
        for child in module.children():
            self._init_parameters(child)

    def log_replacement(self, name: str, old_module: nn.Module, new_module: nn.Module):
        logger.debug("%s: %s -> %s", name, old_module, new_module)

    # -- TP plan handling ---------------------------------------------------
    def _get_model_tp_plan(self) -> Mapping[str, str]:
        plan = (
            getattr(self.model, "tp_plan", None)
            or getattr(self.model, "_tp_plan", None)
            or getattr(self.model.config, "base_model_tp_plan", None)
            or getattr(self.text_config, "base_model_tp_plan", None)
        )
        if plan:
            return plan

        plan = self._infer_tp_plan_from_children()
        return plan if plan else {}

    _LANGUAGE_MODEL_CHILD_NAMES = frozenset(
        {"language_model", "text_model", "model", "lm"}
    )

    def _infer_tp_plan_from_children(self) -> dict[str, str]:
        plan: dict[str, str] = {}
        for child_name, child_module in self.model.named_children():
            child_plan = getattr(child_module, "_tp_plan", None)
            if child_plan:
                plan.update({f"{child_name}.{k}": v for k, v in child_plan.items()})
                continue

            child_config = getattr(child_module, "config", None)
            if child_config is not None:
                child_tp = getattr(child_config, "base_model_tp_plan", None)
                if child_tp:
                    plan.update({f"{child_name}.{k}": v for k, v in child_tp.items()})
                    continue

            if child_name not in self._LANGUAGE_MODEL_CHILD_NAMES:
                continue
            if child_config is None:
                continue
            model_type = getattr(child_config, "model_type", "")
            base_type = (
                model_type.replace("_vl_text", "")
                .replace("_vl", "")
                .replace("_text", "")
            )
            if base_type and base_type != model_type:
                try:
                    from transformers import AutoConfig

                    base_cfg = AutoConfig.for_model(base_type)
                    base_tp = getattr(base_cfg, "base_model_tp_plan", None)
                    if base_tp:
                        plan.update(
                            {f"{child_name}.{k}": v for k, v in base_tp.items()}
                        )
                except Exception as e:
                    logger.debug(
                        "Could not infer TP plan from base model type '%s': %s",
                        base_type,
                        e,
                    )
        return plan

    def _normalize_tp_plan(self, tp_plan: Mapping[str, str]) -> dict[str, Style]:
        normalized = {}
        for pattern, style in tp_plan.items():
            if pattern.startswith("^model\\."):
                pattern = "^" + pattern[len("^model\\.") :]
            elif pattern.startswith("model\\."):
                pattern = pattern[len("model\\.") :]
            elif pattern.startswith("model."):
                pattern = pattern[len("model.") :]
            style = _normalize_tp_style(style)
            if style is None:
                logger.debug(
                    "Dropping TP plan entry '%s' (handled outside the linear "
                    "replacement pass).",
                    pattern,
                )
                continue
            normalized[pattern] = style
        return normalized

    # -- Recursive module replacement (Linear + RMSNorm) --------------------
    def recursive_replace(self):
        tp_size = get_tensor_model_parallel_world_size()
        tp_plan = self._normalize_tp_plan(self._get_model_tp_plan())

        if not tp_plan and tp_size > 1:
            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!"
            )

        # Prefix patterns to match from `self.model`
        prefixed_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                new_module = child_module

                if isinstance(child_module, nn.Linear):
                    pattern = next(
                        (p for p in prefixed_plan if re.match(p, qual_name)),
                        None,
                    )
                    style = prefixed_plan.get(pattern, "replicate")
                    new_module = replace_linear_class(
                        child_module,
                        style,
                        self.quant_config,
                        prefix=qual_name,
                    )
                elif child_module.__class__.__name__.endswith("RMSNorm"):
                    new_module = replace_rms_norm_class(
                        child_module,
                        self.text_config.hidden_size,
                    )
                else:
                    _recursive_replace(child_module, prefix=qual_name)

                if new_module is not child_module:
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)

        _recursive_replace(self.model, prefix="model")

    # -- Pipeline parallel --------------------------------------------------
    def _get_model_pp_plan(self) -> Mapping[str, object]:
        return (
            getattr(self.model, "_pp_plan", None)
            or getattr(self.model, "pp_plan", None)
            or getattr(self.model.config, "base_model_pp_plan", None)
            or getattr(self.text_config, "base_model_pp_plan", None)
            or {}
        )

    def _register_missing_prefix(self, prefix: str):
        if not prefix.endswith("."):
            prefix += "."
        if prefix not in self.skip_prefixes:
            self.skip_prefixes.append(prefix)

    @staticmethod
    def _make_pp_missing_layer(original: nn.Module) -> PPMissingLayer:
        """Create a PPMissingLayer that preserves plain attributes from
        *original* so that the HF forward loop can still access per-layer
        metadata (e.g. ``attention_type`` on Qwen2 decoder layers)."""
        replacement = PPMissingLayer()
        for key, value in original.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (nn.Module, nn.Parameter, torch.Tensor)):
                continue
            setattr(replacement, key, value)
        return replacement

    def _get_submodule_or_none(self, name: str) -> Optional[nn.Module]:
        try:
            return self.model.get_submodule(name)
        except AttributeError:
            return None

    def _set_submodule(self, name: str, module: nn.Module):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = self.model.get_submodule(parent_name)
        else:
            parent_module = self.model
            child_name = name
        setattr(parent_module, child_name, module)

    def pipeline_parallel(self):
        if self.pp_group.world_size <= 1:
            return

        pp_plan = self._get_model_pp_plan()
        if not pp_plan:
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel yet!"
            )

        pp_keys = [re.sub(r"^model\.", "", name) for name in pp_plan.keys()]
        module_list_idx = None
        module_list_name = None
        for idx, name in enumerate(pp_keys):
            if isinstance(self._get_submodule_or_none(name), nn.ModuleList):
                if module_list_idx is not None:
                    raise ValueError(
                        "Pipeline parallel with multiple ModuleList blocks is not supported."
                    )
                module_list_idx = idx
                module_list_name = name

        if module_list_idx is None or module_list_name is None:
            raise ValueError(f"Could not find ModuleList in {type(self.model)}.")

        keep_prefix_modules = self.pp_group.is_first_rank or (
            self.tie_word_embeddings and self.pp_group.is_last_rank
        )
        for name in pp_keys[:module_list_idx]:
            if keep_prefix_modules:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))

        layers = self.model.get_submodule(module_list_name)
        self.start_layer, self.end_layer = get_pp_indices(
            len(layers),
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        for idx in range(len(layers)):
            if self.start_layer <= idx < self.end_layer:
                continue
            layers[idx] = self._make_pp_missing_layer(layers[idx])
            self._register_missing_prefix(
                maybe_prefix("model", f"{module_list_name}.{idx}")
            )

        for name in pp_keys[module_list_idx + 1 :]:
            if self.pp_group.is_last_rank:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))

    # -- Attention instances ------------------------------------------------
    def _create_attention_instances(self, tp_size: int) -> dict[int, RadixAttention]:
        num_heads = self.text_config.num_attention_heads
        num_kv_heads = getattr(self.text_config, "num_key_value_heads", num_heads)
        hidden_size = self.text_config.hidden_size
        head_dim = getattr(self.text_config, "head_dim", hidden_size // num_heads)

        layer_types = getattr(self.text_config, "layer_types", None) or getattr(
            self.config, "layer_types", None
        )
        global_sliding_window = getattr(
            self.text_config, "sliding_window", None
        ) or getattr(self.config, "sliding_window", None)

        if self.is_encoder_only:
            logger.info(
                "Detected encoder-only model (non-causal attention). "
                "Using RadixAttention with AttentionType.ENCODER_ONLY."
            )
        attn_type = (
            AttentionType.ENCODER_ONLY
            if self.is_encoder_only
            else AttentionType.DECODER
        )

        instances = {}
        for idx in range(self.start_layer, self.end_layer):
            # Per-layer sliding window (e.g. Gemma2, Cohere)
            per_layer_sliding_window = -1
            if (
                layer_types is not None
                and idx < len(layer_types)
                and layer_types[idx] == "sliding_attention"
                and global_sliding_window is not None
            ):
                per_layer_sliding_window = global_sliding_window

            instances[idx] = RadixAttention(
                num_heads=divide(num_heads, tp_size),
                head_dim=head_dim,
                scaling=head_dim**-0.5,
                num_kv_heads=divide(num_kv_heads, tp_size),
                layer_id=idx,
                quant_config=self.quant_config,
                sliding_window_size=per_layer_sliding_window,
                attn_type=attn_type,
                prefix=f"{idx}.attn",
            )
        return instances

    # -- Vocab embedding replacement ----------------------------------------
    def replace_vocab_embed_class(self, module: nn.Module):
        old_module = self.model.get_input_embeddings()
        if old_module is None or isinstance(old_module, PPMissingLayer):
            return
        embedding_dim = getattr(old_module, "embedding_dim", None)
        if embedding_dim is None:
            embedding_dim = _getattr_first(
                self.text_config,
                ("embedding_size", "hidden_size"),
                None,
            )
        assert embedding_dim is not None
        new_module = VocabParallelEmbedding(
            self.vocab_size,
            embedding_dim,
            org_num_embeddings=self.vocab_size,
            quant_config=None,
        )

        old_embed_scale = getattr(old_module, "embed_scale", None)
        if old_embed_scale is not None:
            base_cls = new_module.__class__

            class ScaledEmbedding(base_cls):
                def forward(self, input_):
                    return base_cls.forward(self, input_) * self.embed_scale

            new_module.__class__ = ScaledEmbedding
            new_module.embed_scale = old_embed_scale
            self.embed_scale = None

        self.log_replacement("input embedding", old_module, new_module)
        self.model.set_input_embeddings(new_module)

    # -- Forward ------------------------------------------------------------
    def _format_position_ids(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, None, ...]
        if positions.ndim == 1:
            return positions[None, ...]
        return positions

    def _run_hf_backbone(
        self,
        input_ids: Optional[torch.Tensor],
        input_embeds: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        hf_input_ids = None if input_ids is None else input_ids[None, ...]
        hf_input_embeds = None
        if input_embeds is not None:
            hf_input_embeds = input_embeds[None, ...]
            hf_input_ids = None

        # Scale embeddings if needed
        if (
            self.embed_scale is not None
            and hf_input_ids is not None
            and hf_input_embeds is None
        ):
            hf_input_embeds = (
                self.model.get_input_embeddings()(hf_input_ids) * self.embed_scale
            )
            hf_input_ids = None

        if self._position_offset and positions.ndim == 1:
            positions = positions + self._position_offset

        # Cross-encoders distinguish the query/document segments through
        # token type ids; pass them through for models that consume them.
        if (
            self._accepts_token_type_ids
            and "token_type_ids" not in kwargs
            and forward_batch.token_type_ids is not None
        ):
            token_type_ids = forward_batch.token_type_ids
            if token_type_ids.ndim == 1:
                token_type_ids = token_type_ids[None, ...]
            kwargs["token_type_ids"] = token_type_ids

        return self.model(
            input_ids=hf_input_ids,
            inputs_embeds=hf_input_embeds,
            use_cache=False,
            position_ids=self._format_position_ids(positions),
            return_dict=False,
            forward_batch=forward_batch,
            attention_instances=self.attention_instances,
            **kwargs,
        )[0][0, ...]

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._run_hf_backbone(
            input_ids=input_ids,
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> Union[LogitsProcessorOutput, EmbeddingPoolerOutput, PPProxyTensors]:
        runtime_input_ids: Optional[torch.Tensor] = input_ids
        runtime_input_embeds = input_embeds
        if not self.pp_group.is_first_rank:
            assert pp_proxy_tensors is not None
            runtime_input_ids = None
            runtime_input_embeds = pp_proxy_tensors["hidden_states"]

        hidden_states = self._forward_hidden_states(
            input_ids=runtime_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=runtime_input_embeds,
        )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": hidden_states}
            )

        if get_embedding:
            assert (
                self.pooler is not None
            ), "pooling is not enabled for this model class"
            return self._pool(input_ids, hidden_states, forward_batch)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            aux_hidden_states = [
                self._aux_hidden_states[i] for i in self._layers_to_capture
            ]

        assert self.logits_processor is not None and self.lm_head is not None
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def _pool(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        return self.pooler(hidden_states, forward_batch)

    # -- Weight loading -----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )
        return loader.load_weights(weights, mapper=self.weight_mapper)
