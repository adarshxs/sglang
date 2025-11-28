# SPDX-License-Identifier: Apache-2.0
"""
Generic sampling parameters for diffusers backend.

This module provides generic sampling parameters that work with any diffusers pipeline.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.base import DataType, SamplingParams


@dataclass
class DiffusersGenericSamplingParams(SamplingParams):
    """
    Generic sampling parameters for diffusers backend.

    These parameters cover the most common options across different diffusers pipelines.
    The diffusers pipeline will use whichever parameters it supports.

    Note: We don't add diffusers-specific fields here because they get passed
    to Req which doesn't support them. Instead, use the `extra` dict in Req
    for pipeline-specific parameters.
    """

    # Override defaults with more conservative values that work across pipelines
    num_frames: int = 1  # Default to image generation
    height: int = 1024  # Modern models often need 1024x1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = ""  # Empty by default for diffusers compatibility

    def __post_init__(self) -> None:
        # Set data type based on num_frames
        if self.num_frames > 1:
            self.data_type = DataType.VIDEO
        else:
            self.data_type = DataType.IMAGE

        # Don't override width/height if provided
        if self.width is None:
            self.width_not_provided = True
            self.width = 512
        if self.height is None:
            self.height_not_provided = True
            self.height = 512

