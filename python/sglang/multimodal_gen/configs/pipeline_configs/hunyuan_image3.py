from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class HunyuanImage3PipelineConfig(ImagePipelineConfig):
    """Config for HunyuanImage3 text-to-image pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2I
    should_use_guidance: bool = False

    # Avoid VAE/DiT-specific assumptions in base ImagePipelineConfig.
    enable_autocast: bool = False
    vae_tiling: bool = False
    vae_sp: bool = False
