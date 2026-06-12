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
"""Wrapper around `transformers` models."""

from sglang.srt.models.transformers.base import TransformersBase
from sglang.srt.models.transformers.causal import CausalMixin
from sglang.srt.models.transformers.moe import MoEMixin
from sglang.srt.models.transformers.multimodal import MultiModalMixin
from sglang.srt.models.transformers.pooling import (
    EmbeddingMixin,
    SequenceClassificationMixin,
)

# Re-exported for native models that import them from this module.
from sglang.srt.models.transformers.utils import maybe_prefix  # noqa: F401


class TransformersForCausalLM(CausalMixin, TransformersBase):
    pass


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalMoEForCausalLM(
    MultiModalMixin, MoEMixin, CausalMixin, TransformersBase
):
    pass


class TransformersEmbeddingModel(EmbeddingMixin, TransformersBase):
    pass


class TransformersMoEEmbeddingModel(MoEMixin, EmbeddingMixin, TransformersBase):
    pass


class TransformersMultiModalEmbeddingModel(
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEEmbeddingModel(
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersForSequenceClassification(
    SequenceClassificationMixin, TransformersBase
):
    pass


class TransformersMoEForSequenceClassification(
    MoEMixin, SequenceClassificationMixin, TransformersBase
):
    pass


class TransformersMultiModalForSequenceClassification(
    MultiModalMixin, SequenceClassificationMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEForSequenceClassification(
    MultiModalMixin, MoEMixin, SequenceClassificationMixin, TransformersBase
):
    pass


EntryClass = [
    TransformersForCausalLM,
    TransformersMoEForCausalLM,
    TransformersMultiModalForCausalLM,
    TransformersMultiModalMoEForCausalLM,
    TransformersEmbeddingModel,
    TransformersMoEEmbeddingModel,
    TransformersMultiModalEmbeddingModel,
    TransformersMultiModalMoEEmbeddingModel,
    TransformersForSequenceClassification,
    TransformersMoEForSequenceClassification,
    TransformersMultiModalForSequenceClassification,
    TransformersMultiModalMoEForSequenceClassification,
]
