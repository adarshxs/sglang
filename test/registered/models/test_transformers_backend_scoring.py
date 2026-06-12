# Copyright 2023-2024 SGLang Team
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
"""Transformers-backend (``--model-impl transformers``) scoring coverage.

Exercises ``TransformersForSequenceClassification`` for the score-producing
model families, comparing against SGLang's native implementations
(``model_impl="auto"``), mirroring the reference choice of the sibling
``test_transformers_backend_embedding.py``:

- encoder cross-encoders / rerankers via the rerank (score) API:
  ``BertForSequenceClassification`` (base-model pooler + linear classifier)
  and ``XLMRobertaForSequenceClassification`` (classification head, offset
  position ids);
- decoder sequence-classification / reward models via the encode API
  (``Qwen3ForSequenceClassification`` with a last-token score head).
"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS, SRTRunner
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=420, suite="stage-b-test-1-gpu-small")

# (model, tp_size, score_tolerance); float32 + triton, matching the native
# cross-encoder test (test_cross_encoder_models.py).
CROSS_ENCODER_MODELS = [
    ("cross-encoder/ms-marco-MiniLM-L6-v2", 1, 1e-2),
    ("BAAI/bge-reranker-v2-m3", 1, 1e-2),
]

# (model, tp_size, score_tolerance); fp16. The tolerance mirrors the native
# reward test (test_reward_models.py) for this checkpoint.
REWARD_MODELS = [
    ("Skywork/Skywork-Reward-V2-Qwen3-0.6B", 1, 1e-1),
]

REWARD_CONVS = [
    [
        {
            "role": "user",
            "content": "What is the range of the numeric output of a "
            "sigmoid node in a neural network?",
        },
        {
            "role": "assistant",
            "content": "The output of a sigmoid node is bounded between -1 and 1.",
        },
    ],
    [
        {
            "role": "user",
            "content": "What is the range of the numeric output of a "
            "sigmoid node in a neural network?",
        },
        {
            "role": "assistant",
            "content": "The output of a sigmoid node is bounded between 0 and 1.",
        },
    ],
]


class TestTransformersBackendScoring(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _rerank_scores(self, model_path, model_impl, tp_size, torch_dtype, pairs):
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="cross_encoder",
            model_impl=model_impl,
            attention_backend="triton",
            chunked_prefill_size=-1,
            disable_radix_cache=True,
        ) as runner:
            return runner.forward(pairs).scores

    def test_cross_encoder_scores(self):
        torch_dtype = torch.float32
        for model_path, tp_size, tolerance in CROSS_ENCODER_MODELS:
            for query_docs in TEST_RERANK_QUERY_DOCS:
                pairs = [
                    [query_docs["query"], document]
                    for document in query_docs["documents"]
                ]
                transformers_scores = self._rerank_scores(
                    model_path, "transformers", tp_size, torch_dtype, pairs
                )
                native_scores = self._rerank_scores(
                    model_path, "auto", tp_size, torch_dtype, pairs
                )
                for i, (transformers_score, native_score) in enumerate(
                    zip(transformers_scores, native_scores)
                ):
                    difference = abs(transformers_score - native_score)
                    print(
                        f"[{model_path}] pair {i}: transformers={transformers_score:.5f} "
                        f"native={native_score:.5f} diff={difference:.3e}"
                    )
                    assert difference < tolerance, (
                        f"transformers-backend rerank score not close to native "
                        f"for {model_path} (pair {i})"
                    )

    def _reward_scores(self, model_path, model_impl, tp_size, torch_dtype, prompts):
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="reward",
            model_impl=model_impl,
        ) as runner:
            return runner.forward(prompts).scores

    def test_reward_scores(self):
        from transformers import AutoTokenizer

        torch_dtype = torch.float16
        for model_path, tp_size, tolerance in REWARD_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            prompts = tokenizer.apply_chat_template(REWARD_CONVS, tokenize=False)
            transformers_scores = self._reward_scores(
                model_path, "transformers", tp_size, torch_dtype, prompts
            )
            native_scores = self._reward_scores(
                model_path, "auto", tp_size, torch_dtype, prompts
            )
            for i, (transformers_score, native_score) in enumerate(
                zip(transformers_scores, native_scores)
            ):
                difference = abs(transformers_score - native_score)
                print(
                    f"[{model_path}] conv {i}: transformers={transformers_score:.5f} "
                    f"native={native_score:.5f} diff={difference:.3e}"
                )
                assert difference < tolerance, (
                    f"transformers-backend reward score not close to native "
                    f"for {model_path} (conv {i})"
                )


if __name__ == "__main__":
    unittest.main()
