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
"""Transformers modeling backend mixin for causal language models."""

from typing import List, Optional

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.transformers.utils import maybe_prefix


class CausalMixin:

    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)

        if self.tie_word_embeddings:
            self.skip_prefixes.append("lm_head.")

        if not self.pp_group.is_last_rank:
            self._register_missing_prefix("lm_head")
            return

        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        logit_scale = getattr(self.text_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.text_config, logit_scale=logit_scale
        )

    def get_embed_and_head(self):
        """Expose the input embedding and lm_head weights (shared with
        speculative draft models, e.g. EAGLE)."""
        return self.model.get_input_embeddings().weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        """Capture aux hidden states for EAGLE3 draft models, matching the
        native models' semantics: the inputs of layers [2, n/2, n-3] by
        default, or of layers ``layer_id + 1`` for explicit ids."""
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            num_layers = self.text_config.num_hidden_layers
            layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            layers_to_capture = [layer_id + 1 for layer_id in layer_ids]
        self._install_aux_hidden_state_hooks(layers_to_capture)
        self.capture_aux_hidden_states = True
