# SPDX-License-Identifier: Apache-2.0
"""
Generic pipeline configuration for diffusers backend.

This module provides a minimal pipeline configuration that works with the diffusers backend.
Since diffusers handles its own model loading and configuration, this config is intentionally minimal.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import (
    DiTConfig,
    EncoderConfig,
    VAEConfig,
)
from sglang.multimodal_gen.configs.pipelines.base import ModelTaskType, PipelineConfig


@dataclass
class DiffusersGenericPipelineConfig(PipelineConfig):
    """
    Generic pipeline configuration for diffusers backend.

    This is a minimal configuration since the diffusers backend handles most
    configuration internally. It provides sensible defaults for the required fields.
    """

    # Default to T2I since it's the most common use case
    task_type: ModelTaskType = ModelTaskType.T2I

    # Precision settings - diffusers will use these
    dit_precision: str = "fp16"
    vae_precision: str = "fp32"

    # Disable sglang-specific features that don't apply to diffusers backend
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 1.0
    flow_shift: float | None = None
    disable_autocast: bool = True  # Let diffusers handle dtype

    # Minimal model configs - diffusers handles its own loading
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

    # VAE settings
    vae_tiling: bool = False  # Disable by default, diffusers handles this
    vae_sp: bool = False

    def check_pipeline_config(self) -> None:
        """
        Override to skip most validation since diffusers handles its own config.
        """
        # Only do minimal validation
        pass

    def adjust_size(self, width, height, image):
        """
        Pass through - diffusers handles size adjustments.
        """
        return width, height

    def adjust_num_frames(self, num_frames):
        """
        Pass through - diffusers handles frame count.
        """
        return num_frames

