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
"""Transformers modeling backend mixins for pooling models."""

import json
import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from sglang.srt.layers.pooler import (
    CrossEncodingPooler,
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.transformers.utils import _getattr_first

logger = logging.getLogger(__name__)


_ST_POOLING_MODE_TO_TYPE = {
    "pooling_mode_cls_token": PoolingType.CLS,
    "pooling_mode_mean_tokens": PoolingType.MEAN,
    "pooling_mode_lasttoken": PoolingType.LAST,
}


def _load_repo_json(model_path: str, file_name: str) -> Optional[Union[dict, list]]:
    """Load a JSON file from a model repo (local dir or hub cache), or None."""
    try:
        from transformers.utils import cached_file

        resolved = cached_file(
            model_path,
            file_name,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved is None:
            return None
        with open(resolved) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Could not load %s from %s: %s", file_name, model_path, e)
        return None


def get_sentence_transformers_pooling(
    model_path: str,
) -> tuple[Optional[PoolingType], Optional[bool]]:
    """Read the pooling type and normalization from a checkpoint's
    sentence-transformers configuration (``modules.json``), if present.

    Returns ``(pooling_type, normalize)``.  Both are None when the repo has
    no sentence-transformers configuration; ``pooling_type`` alone is None
    when the configured pooling mode has no SGLang equivalent.
    """
    modules = _load_repo_json(model_path, "modules.json")
    if not isinstance(modules, list):
        return None, None

    pooling_type = None
    normalize = False
    for module in modules:
        module_type = module.get("type", "")
        if module_type == "sentence_transformers.models.Pooling":
            pooling_config = _load_repo_json(
                model_path, f"{module.get('path')}/config.json"
            )
            if not isinstance(pooling_config, dict):
                continue
            enabled_modes = [
                mode
                for mode in pooling_config
                if mode.startswith("pooling_mode_") and pooling_config[mode]
            ]
            pooling_type = next(
                (
                    _ST_POOLING_MODE_TO_TYPE[mode]
                    for mode in enabled_modes
                    if mode in _ST_POOLING_MODE_TO_TYPE
                ),
                None,
            )
            if pooling_type is None:
                logger.warning(
                    "The sentence-transformers pooling mode(s) %s of %s are "
                    "not supported by the Transformers backend; falling back "
                    "to the default pooling type.",
                    enabled_modes,
                    model_path,
                )
        elif module_type == "sentence_transformers.models.Normalize":
            normalize = True
    return pooling_type, normalize


class EmbeddingMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_unexpected_prefixes.append("lm_head.")
        if not self.pp_group.is_last_rank:
            return
        pooling_type, normalize = self._resolve_pooling_config()
        logger.info(
            "Transformers backend pooling: pooling_type=%s normalize=%s",
            pooling_type.name,
            normalize,
        )
        self.pooler = Pooler(pooling_type=pooling_type, normalize=normalize)

    def _resolve_pooling_config(self) -> Tuple[PoolingType, bool]:
        """Resolve the pooling type and normalization for this checkpoint.

        Lowest to highest priority: encoder/decoder defaults matching
        SGLang's native pooling models, the checkpoint's
        sentence-transformers configuration, and explicit ``pooling_type`` /
        ``normalize`` config attributes (settable through
        ``--json-model-override-args``).
        """
        # Encoder-only models pool the CLS token, decoders the last token,
        # matching SGLang's native BertModel / Qwen-family embedding models.
        pooling_type = PoolingType.CLS if self.is_encoder_only else PoolingType.LAST
        normalize = True

        st_pooling_type, st_normalize = get_sentence_transformers_pooling(
            getattr(self.config, "_name_or_path", "")
        )
        if st_pooling_type is not None:
            pooling_type = st_pooling_type
        if st_normalize is not None:
            normalize = st_normalize

        config_pooling = getattr(self.config, "pooling_type", None)
        if config_pooling is not None:
            try:
                pooling_type = PoolingType[str(config_pooling).upper()]
            except KeyError:
                supported = ", ".join(p.name for p in PoolingType)
                raise ValueError(
                    f"Unknown pooling_type '{config_pooling}'; supported "
                    f"types are: {supported}."
                ) from None
        config_normalize = getattr(self.config, "normalize", None)
        if config_normalize is not None:
            normalize = bool(config_normalize)

        return pooling_type, normalize


class SequenceClassificationMixin:
    """Serve ``*ForSequenceClassification`` checkpoints (sequence
    classification, cross-encoding/reranking, and decoder reward models).

    The classification head is discovered by instantiating the checkpoint's
    ``AutoModelForSequenceClassification`` class on the meta device.  Scoring
    then follows SGLang's native conventions:

    - encoder-only models (BERT/RoBERTa-family cross-encoders) reproduce
      ``BertForSequenceClassification``: the base model's pooler (dense +
      tanh on the CLS token) when present, the classification head, and the
      checkpoint's sentence-transformers activation, via
      ``CrossEncodingPooler``;
    - decoder models reproduce ``Qwen2ForSequenceClassification``: the score
      head pooled at the last token via ``score_and_pool`` (raw logits, no
      activation), which also provides multi-item scoring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_unexpected_prefixes.append("lm_head.")
        if not self.pp_group.is_last_rank:
            return

        with torch.device("meta"):
            seq_cls_model = AutoModelForSequenceClassification.from_config(
                self.config,
                torch_dtype=torch.get_default_dtype(),
                trust_remote_code=True,
            )

        # Some architectures drop the base model's pooler for sequence
        # classification (e.g. RoBERTa); mirror that so we do not keep
        # parameters the checkpoint does not have.
        for module in seq_cls_model.modules():
            if hasattr(module, "pooler") and module.pooler is None:
                self.model.pooler = None
                break

        classifier = _getattr_first(seq_cls_model, ("classifier", "score"))
        if classifier is None:
            raise ValueError(
                f"Could not find a 'classifier' or 'score' layer in "
                f"{type(seq_cls_model).__name__}, so this checkpoint cannot "
                "be served through TransformersForSequenceClassification."
            )
        self._init_parameters(classifier)
        self.classifier = classifier

        if self.is_encoder_only:
            hf_pooler = getattr(self.model, "pooler", None)
            if hf_pooler is not None:
                # The classifier consumes the pooler's CLS representation,
                # so it can be applied to the stacked per-request outputs.
                per_request_pooler = self._adapt_per_request(hf_pooler)
                self.pooler = CrossEncodingPooler(
                    self.config, classifier, per_request_pooler
                )
            else:
                # The classification head consumes the token-level hidden
                # states directly (e.g. RobertaClassificationHead).
                per_request_classifier = self._adapt_per_request(classifier)
                self.pooler = CrossEncodingPooler(self.config, per_request_classifier)
        else:
            self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @staticmethod
    def _adapt_per_request(module: nn.Module):
        """Adapt a batch-first Transformers head to ``CrossEncodingPooler``'s
        per-request convention ([seq_len, hidden] in, no batch dim out)."""

        def adapted(hidden_states: torch.Tensor, *args) -> torch.Tensor:
            return module(hidden_states.unsqueeze(0)).squeeze(0)

        return adapted

    def _pool(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        if self.is_encoder_only:
            return self.pooler(hidden_states, forward_batch)
        return score_and_pool(
            self.classifier, self.pooler, hidden_states, forward_batch, input_ids
        )
