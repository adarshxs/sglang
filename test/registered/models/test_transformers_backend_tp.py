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
"""Transformers-backend (``--model-impl transformers``) tensor-parallel
coverage (tp_size=2).

Checks that the backend's tensor-parallel module replacement (the
Transformers ``base_model_tp_plan``, including the v5
``colwise_gather_output`` / ``rowwise_split_input`` styles) produces the same
results as SGLang's native implementations at tp_size=2:

- generation (``Qwen/Qwen2.5-0.5B-Instruct``): prefill logprobs vs native;
- embedding (``Qwen/Qwen3-Embedding-0.6B``): pooled embeddings vs native.

Models whose plans gather the attention projections (OLMoE, Phi3-family) are
covered by the local TP matrix in the parity report; their gathered q/k/v are
rank-sliced inside ``sglang_flash_attention_forward`` so attention and the KV
cache stay head-sharded.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_cuda_ci(est_time=420, suite="stage-b-test-2-gpu-large")

TP_SIZE = 2
PROMPTS = [
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
]

GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_LOGPROB_TOLERANCE = 5e-2

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_COSINE_TOLERANCE = 1e-2


class TestTransformersBackendTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _generation_logprobs(self, model_impl):
        with SRTRunner(
            GENERATION_MODEL,
            tp_size=TP_SIZE,
            torch_dtype=torch.float16,
            model_type="generation",
            model_impl=model_impl,
        ) as runner:
            out = runner.forward(PROMPTS, max_new_tokens=8)
        return [torch.tensor(x[-1][:5]) for x in out.top_input_logprobs]

    def test_generation_tp2(self):
        transformers_logprobs = self._generation_logprobs("transformers")
        native_logprobs = self._generation_logprobs("auto")
        for i in range(len(PROMPTS)):
            difference = (
                (transformers_logprobs[i] - native_logprobs[i]).abs().max().item()
            )
            print(
                f"[{GENERATION_MODEL}] prompt {i}: logprob max|diff| = {difference:.4f}"
            )
            assert difference < GENERATION_LOGPROB_TOLERANCE, (
                f"transformers-backend tp2 prefill logprobs not close to native "
                f"for {GENERATION_MODEL} (prompt {i})"
            )

    def _embeddings(self, model_impl):
        with SRTRunner(
            EMBEDDING_MODEL,
            tp_size=TP_SIZE,
            torch_dtype=torch.float16,
            model_type="embedding",
            model_impl=model_impl,
        ) as runner:
            return [torch.tensor(e) for e in runner.forward(PROMPTS).embed_logits]

    def test_embedding_tp2(self):
        transformers_embeddings = self._embeddings("transformers")
        native_embeddings = self._embeddings("auto")
        for i in range(len(PROMPTS)):
            similarity = torch.tensor(
                get_similarities(transformers_embeddings[i], native_embeddings[i])
            )
            print(
                f"[{EMBEDDING_MODEL}] prompt {i}: cosine-sim diff = "
                f"{float(abs(similarity - 1)):.3e}"
            )
            assert torch.all(
                abs(similarity - 1) < EMBEDDING_COSINE_TOLERANCE
            ), f"transformers-backend tp2 embeddings not close to native (prompt {i})"


if __name__ == "__main__":
    unittest.main()
