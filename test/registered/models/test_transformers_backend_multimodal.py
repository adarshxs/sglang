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
"""Transformers-backend (``--model-impl transformers``) multimodal embedding
coverage.

Forces ``TransformersMultiModalEmbeddingModel`` for a small vision-language
embedding checkpoint (``marco/mcdse-2b-v1``, Qwen2-VL-based, also used by the
native ``test/registered/embedding/test_embedding_models.py``) and checks both
text-only and image embeddings stay close to SGLang's native implementation
(``model_impl="auto"``), mirroring the reference choice of the sibling
``test_transformers_backend_embedding.py``.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_cuda_ci(est_time=380, suite="stage-b-test-1-gpu-small")

MODEL = "marco/mcdse-2b-v1"
TOLERANCE = 1e-2

TEXT_PROMPTS = [
    "The capital of the United Kingdom is London.",
    "Today is a sunny day and I like to walk in the park.",
]
IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image?<|im_end|>\n"
)
IMAGE_URL = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"


class TestTransformersBackendMultimodalEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _forward_embeddings(self, model_impl):
        """Return (text_embeddings, image_embedding) for one impl."""
        with SRTRunner(
            MODEL,
            tp_size=1,
            torch_dtype=torch.float16,
            model_type="embedding",
            model_impl=model_impl,
        ) as runner:
            text_logits = runner.forward(TEXT_PROMPTS).embed_logits
            image_logits = runner.forward(
                [IMAGE_PROMPT], image_data=[IMAGE_URL]
            ).embed_logits
        return text_logits, image_logits

    def test_multimodal_embedding(self):
        transformers_text, transformers_image = self._forward_embeddings("transformers")
        native_text, native_image = self._forward_embeddings("auto")

        cases = [
            *(
                (f"text prompt {i}", transformers_text[i], native_text[i])
                for i in range(len(TEXT_PROMPTS))
            ),
            ("image prompt", transformers_image[0], native_image[0]),
        ]
        for tag, transformers_emb, native_emb in cases:
            similarity = torch.tensor(
                get_similarities(
                    torch.tensor(transformers_emb), torch.tensor(native_emb)
                )
            )
            print(
                f"[{MODEL}] {tag}: cosine-sim diff = {float(abs(similarity - 1)):.3e}"
            )
            assert torch.all(
                abs(similarity - 1) < TOLERANCE
            ), f"transformers-backend embeddings not close to native for {MODEL} ({tag})"


if __name__ == "__main__":
    unittest.main()
