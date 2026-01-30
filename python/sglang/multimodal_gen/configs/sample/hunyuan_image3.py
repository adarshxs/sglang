from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    """Sampling defaults for HunyuanImage3 (image-only)."""

    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    negative_prompt: str = ""

    # Optional fields reserved for future expansion.
    image_size: str | None = None
    bot_task: str = "image"
    task_extra_kwargs: dict[str, Any] = field(default_factory=dict)
