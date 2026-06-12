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
"""Transformers modeling backend mixin for multimodal models."""

import inspect
from collections.abc import Mapping
from typing import Optional

import torch

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.utils import WeightsMapper


def _encoder_accepts_feature_kwarg(encoder, feature_kwarg: str) -> bool:
    try:
        sig = inspect.signature(encoder)
    except (TypeError, ValueError):
        return False

    if feature_kwarg in sig.parameters:
        return True

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_keyword:
        return False

    required_positional_params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(required_positional_params) == 0


_MULTIMODAL_DYNAMIC_ARG_DIMS: dict[str, int] = {
    "input_ids": 0,
    "positions": -1,  # last dim to support M-RoPE (Qwen2.5-VL 3×seq layout)
    "input_embeds": 0,
}


class MultiModalMixin:
    torch_compile_dynamic_arg_dims: dict[str, int] = _MULTIMODAL_DYNAMIC_ARG_DIMS

    # Older VL checkpoints (e.g. Qwen2.5-VL) store text weights as
    # "model.layers.*" but transformers >=5.0 nests the text model under
    # "model.language_model.*".  Map explicitly so these load correctly.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",
            "text_model.model.": "model.text_model.",
            "text_model.lm_head.": "lm_head.",
            "language_model.lm_head.": "lm_head.",
            "vision_tower.": "model.vision_tower.",
            "vision_model.": "model.vision_model.",
            "vision_embed_tokens.": "model.vision_embed_tokens.",
            "image_newline.": "model.image_newline.",
            "vqmodel.": "model.vqmodel.",
            "multi_modal_projector.": "model.multi_modal_projector.",
            "visual.": "model.visual.",
            "model.layers.": "model.language_model.layers.",
            "model.embed_tokens.": "model.language_model.embed_tokens.",
            "model.norm.": "model.language_model.norm.",
            "model.rotary_emb.": "model.language_model.rotary_emb.",
        }
    )

    _mm_feature_kwarg = {
        "image": "pixel_values",
        "video": "pixel_values_videos",
        "audio": "input_features",
    }
    _mm_encoder_candidates = {
        "image": ("get_image_features", "get_image_feature"),
        "video": ("get_video_features", "get_video_feature"),
        "audio": ("get_audio_features", "get_audio_feature"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mm_padding_pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def _uses_mrope_positions(self) -> bool:
        rope_scaling = getattr(self.text_config, "rope_scaling", None)
        if isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling:
            return True
        rope_type = str(getattr(self.text_config, "rope_type", "")).lower()
        return "mrope" in rope_type

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        return input_ids

    def _get_modality_encoder(self, modality_name: str):
        for name in self._mm_encoder_candidates[modality_name]:
            fn = getattr(self.model, name, None)
            if fn is not None:
                return fn
        raise AttributeError(f"No encoder method found for modality '{modality_name}'")

    def _get_modality_dtype_device(
        self, modality_name: str
    ) -> tuple[Optional[torch.dtype], Optional[torch.device]]:
        module_candidates = {
            "image": ("vision_tower", "vision_model"),
            "video": ("video_tower", "vision_tower", "vision_model"),
            "audio": ("audio_tower", "audio_model", "audio_encoder"),
        }
        modules = []
        for name in module_candidates.get(modality_name, ()):
            module = getattr(self.model, name, None)
            if module is not None:
                modules.append(module)
        modules.append(self.model)

        for module in modules:
            for param in module.parameters():
                if torch.is_floating_point(param):
                    return param.dtype, param.device
            for buf in module.buffers():
                if torch.is_floating_point(buf):
                    return buf.dtype, buf.device
        return None, None

    def _cast_mm_value(self, value, dtype, device):
        if torch.is_tensor(value):
            if value.is_floating_point() and dtype is not None:
                return value.to(dtype=dtype, device=device)
            return value
        if isinstance(value, dict):
            return {k: self._cast_mm_value(v, dtype, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_mm_value(v, dtype, device) for v in value]
        if isinstance(value, tuple):
            return tuple(self._cast_mm_value(v, dtype, device) for v in value)
        return value

    def _to_tensor_output(self, output) -> torch.Tensor:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            output = output.pooler_output
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Empty multimodal encoder output.")
            if all(torch.is_tensor(x) for x in output):
                output = torch.cat(
                    [x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x for x in output],
                    dim=0,
                )
            else:
                output = output[0]
        elif hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, dict):
            if output.get("pooler_output", None) is not None:
                output = output["pooler_output"]
            else:
                output = next(v for v in output.values() if torch.is_tensor(v))
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Empty multimodal encoder output.")
                if all(torch.is_tensor(x) for x in output):
                    output = torch.cat(
                        [
                            x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                            for x in output
                        ],
                        dim=0,
                    )
                else:
                    output = output[0]

        if output.ndim > 2:
            output = output.reshape(-1, output.shape[-1])
        return output

    def _encode_modality_items(
        self, modality_name: str, items: list[MultimodalDataItem]
    ) -> torch.Tensor:
        encoder = self._get_modality_encoder(modality_name)
        feature_kwarg = self._mm_feature_kwarg[modality_name]
        target_dtype, target_device = self._get_modality_dtype_device(modality_name)
        outputs = []
        for item in items:
            kwargs = self._cast_mm_value(
                dict(item.model_specific_data),
                dtype=target_dtype,
                device=target_device,
            )
            feature = self._cast_mm_value(
                item.feature,
                dtype=target_dtype,
                device=target_device,
            )
            if _encoder_accepts_feature_kwarg(encoder, feature_kwarg):
                kwargs[feature_kwarg] = feature
                result = encoder(**kwargs)
            else:
                result = encoder(feature, **kwargs)
            outputs.append(self._to_tensor_output(result))
        return torch.cat(outputs, dim=0)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("image", items)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("video", items)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("audio", items)

    def _collect_mm_kwargs(self, forward_batch: ForwardBatch) -> dict:
        """Collect multimodal tensors from the forward batch and return them
        as kwargs suitable for the HF model's forward method."""
        kwargs = {}

        if getattr(forward_batch, "token_type_ids", None) is not None:
            tti = forward_batch.token_type_ids
            if tti.ndim == 1:
                tti = tti.unsqueeze(0)
            token_type_key = (
                "mm_token_type_ids"
                if "mm_token_type_ids"
                in inspect.signature(self.model.forward).parameters
                else "token_type_ids"
            )
            kwargs[token_type_key] = tti

        if (
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs = forward_batch.mm_inputs
            target_device = next(self.model.parameters()).device

            for batch_idx in range(len(mm_inputs or [])):
                mm_input = mm_inputs[batch_idx]
                if mm_input is None:
                    continue
                for item in mm_input.mm_items or []:
                    for key, value in (item.model_specific_data or {}).items():
                        if isinstance(value, torch.Tensor):
                            value = value.to(device=target_device)
                        if key not in kwargs:
                            kwargs[key] = value
                        elif isinstance(value, torch.Tensor) and isinstance(
                            kwargs[key], torch.Tensor
                        ):
                            kwargs[key] = torch.cat([kwargs[key], value], dim=0)
                    if item.feature is not None:
                        feature_key = self._mm_feature_kwarg.get(
                            item.modality.name.lower(), "pixel_values"
                        )
                        feature = item.feature
                        if isinstance(feature, torch.Tensor):
                            feature = feature.to(device=target_device)
                        if feature_key not in kwargs:
                            kwargs[feature_key] = feature
                        elif isinstance(feature, torch.Tensor) and isinstance(
                            kwargs[feature_key], torch.Tensor
                        ):
                            kwargs[feature_key] = torch.cat(
                                [kwargs[feature_key], feature], dim=0
                            )

        return kwargs

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            return super()._forward_hidden_states(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        if (
            self._uses_mrope_positions()
            and getattr(forward_batch, "mrope_positions", None) is not None
        ):
            positions = forward_batch.mrope_positions

        mm_kwargs = self._collect_mm_kwargs(forward_batch)

        return self._run_hf_backbone(
            input_ids=input_ids,
            input_embeds=None,
            positions=positions,
            forward_batch=forward_batch,
            **mm_kwargs,
        )
