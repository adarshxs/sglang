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
"""Transformers-backend (``--model-impl transformers``) embedding coverage.

The pre-existing Transformers-backend tests only cover text *generation*
(``test_transformers_backend_eval.py`` and ``test_transformers_models.py``).
This file adds the missing **embedding** coverage: it forces SGLang's generic
Transformers modeling backend (``model_impl="transformers"`` ->
``TransformersEmbeddingModel``) and checks the pooled embeddings stay close to
SGLang's native model implementation, across the pooling configurations the
backend resolves from the checkpoint:

- decoder, last-token pooling + normalize (``Qwen/Qwen3-Embedding-0.6B``),
  including per-request matryoshka dimension truncation;
- encoder, CLS pooling + normalize (``BAAI/bge-small-en``);
- encoder, mean pooling (``sentence-transformers/all-MiniLM-L6-v2``), checked
  against a manual Transformers reference because SGLang's native BERT only
  implements CLS pooling.

Reference choice — this mirrors ``test_transformers_models.py``, the existing
Transformers-backend test, which compares the backend against the native SGLang
implementation (``model_impl="auto"``) rather than against an external library.
The native embedding path is itself validated against HF / sentence-transformers
in ``test/registered/embedding/test_embedding_models.py``, so it is a trustworthy
reference, and -- being independent of the ``transformers`` modeling code -- it
makes this a sharp signal for the nightly job that runs against ``transformers``
``main``: if a ``transformers`` change perturbs the backend's pooled output, it
diverges from the (unaffected) native output and the test fails.

Sequence-classification, cross-encoder, and reward coverage lives in the
sibling ``test_transformers_backend_scoring.py``; multimodal embedding in
``test_transformers_backend_multimodal.py``.
"""

import multiprocessing as mp
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import DEFAULT_PROMPTS, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_cuda_ci(est_time=400, suite="stage-b-test-1-gpu-small")

# (model, tp_size, cosine_tolerance, attention_backend, matryoshka_dim):
# assert abs(cosine_similarity - 1) < tolerance. Both sides run the same
# weights; the only difference is native vs. Transformers modeling, so a tight
# cosine bound catches real breakage while tolerating fp16 / attention-backend
# numerics. Use ungated, small models so the test is fast/free. A non-None
# matryoshka_dim additionally compares embeddings truncated to that dimension.
MODELS = [
    # Decoder family: last-token pooling + normalize (sentence-transformers
    # config), matryoshka truncation.
    ("Qwen/Qwen3-Embedding-0.6B", 1, 1e-2, None, 64),
    # Encoder family (BERT): CLS pooling + normalize.
    ("BAAI/bge-small-en", 1, 1e-2, "triton", None),
]

# Encoder checkpoint whose sentence-transformers config selects mean pooling.
# SGLang's native BERT only implements CLS pooling, so the backend is compared
# against a manual Transformers forward instead of model_impl="auto".
MEAN_POOLING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TORCH_DTYPES = [torch.float16]


class TestTransformersBackendEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", 2048) - 20
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        truncated = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated.append(
                    tokenizer.decode(
                        tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                    )
                )
            else:
                truncated.append(prompt)
        return truncated

    def _forward_embeddings(
        self,
        prompts,
        model_path,
        model_impl,
        tp_size,
        torch_dtype,
        attention_backend,
        matryoshka_dim,
    ):
        """Return (embeddings, matryoshka_embeddings or None) for one impl."""
        json_model_override_args = (
            {"is_matryoshka": True} if matryoshka_dim is not None else None
        )
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            model_impl=model_impl,
            attention_backend=attention_backend,
            json_model_override_args=json_model_override_args,
        ) as runner:
            embeddings = runner.forward(prompts).embed_logits
            matryoshka_embeddings = None
            if matryoshka_dim is not None:
                matryoshka_embeddings = runner.forward(
                    prompts, dimensions=matryoshka_dim
                ).embed_logits
        return embeddings, matryoshka_embeddings

    def _assert_close(
        self, prompts, transformers_logits, native_logits, tolerance, tag
    ):
        for i in range(len(prompts)):
            transformers_emb = torch.tensor(transformers_logits[i])
            native_emb = torch.tensor(native_logits[i])
            similarity = torch.tensor(get_similarities(transformers_emb, native_emb))
            print(
                f"[{tag}] prompt {i}: cosine-sim diff = "
                f"{float(abs(similarity - 1)):.3e}"
            )
            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < tolerance
                ), f"transformers-backend embeddings not close to reference for {tag} (prompt {i})"

    def assert_close_embeddings(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        tolerance,
        attention_backend,
        matryoshka_dim,
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        # Under test: SGLang generic Transformers modeling backend.
        transformers_logits, transformers_matryoshka = self._forward_embeddings(
            truncated_prompts,
            model_path,
            "transformers",
            tp_size,
            torch_dtype,
            attention_backend,
            matryoshka_dim,
        )

        # Reference: SGLang native modeling implementation.
        native_logits, native_matryoshka = self._forward_embeddings(
            truncated_prompts,
            model_path,
            "auto",
            tp_size,
            torch_dtype,
            attention_backend,
            matryoshka_dim,
        )

        self._assert_close(
            truncated_prompts, transformers_logits, native_logits, tolerance, model_path
        )
        if matryoshka_dim is not None:
            for embeddings in (transformers_matryoshka, native_matryoshka):
                for emb in embeddings:
                    assert len(emb) == matryoshka_dim
            self._assert_close(
                truncated_prompts,
                transformers_matryoshka,
                native_matryoshka,
                tolerance,
                f"{model_path} (matryoshka dim={matryoshka_dim})",
            )

    def test_embedding_prefill_logits(self):
        for model, tp_size, tolerance, attention_backend, matryoshka_dim in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_embeddings(
                    DEFAULT_PROMPTS,
                    model,
                    tp_size,
                    torch_dtype,
                    tolerance,
                    attention_backend,
                    matryoshka_dim,
                )

    def test_mean_pooling(self):
        """Mean pooling resolved from the sentence-transformers config."""
        torch_dtype = torch.float16
        tolerance = 1e-2
        prompts = self._truncate_prompts(DEFAULT_PROMPTS, MEAN_POOLING_MODEL)

        transformers_logits, _ = self._forward_embeddings(
            prompts, MEAN_POOLING_MODEL, "transformers", 1, torch_dtype, "triton", None
        )

        # Manual reference: mean over the last hidden states + L2 normalize,
        # the semantics of the checkpoint's sentence-transformers config.
        from transformers import AutoModel

        tokenizer = AutoTokenizer.from_pretrained(MEAN_POOLING_MODEL)
        model = (
            AutoModel.from_pretrained(MEAN_POOLING_MODEL, dtype=torch_dtype)
            .to("cuda")
            .eval()
        )
        reference_logits = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state[0]
            emb = torch.nn.functional.normalize(hidden.mean(dim=0), p=2, dim=-1)
            reference_logits.append(emb.float().cpu().tolist())
        del model
        torch.cuda.empty_cache()

        self._assert_close(
            prompts,
            transformers_logits,
            reference_logits,
            tolerance,
            f"{MEAN_POOLING_MODEL} (mean pooling vs manual HF)",
        )


if __name__ == "__main__":
    unittest.main()
