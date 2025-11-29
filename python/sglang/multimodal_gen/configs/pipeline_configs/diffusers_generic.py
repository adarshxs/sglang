# SPDX-License-Identifier: Apache-2.0
"""
Generic pipeline configuration for diffusers backend.

This module provides a pipeline configuration that works with the diffusers backend
and enables SGLang optimizations.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class DiffusersGenericPipelineConfig(PipelineConfig):
    """
    Generic pipeline configuration for diffusers backend with SGLang optimizations.

    This configuration enables various SGLang optimizations on top of diffusers pipelines:
    - torch.compile for transformer/unet acceleration
    - VAE tiling/slicing for memory efficiency
    - Attention backend replacement (FlashAttention, SageAttention, etc.)
    - CPU offloading for memory-constrained systems

    Example usage:
        ```python
        config = DiffusersGenericPipelineConfig(
            vae_tiling=True,
            attention_slicing=True,
            use_sglang_attention=True,
            attention_backend="flash_attn",
        )
        ```
    """

    # default to T2I since it's the most common
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"

    should_use_guidance: bool = True
    embedded_cfg_scale: float = 1.0
    flow_shift: float | None = None
    disable_autocast: bool = True  # let diffusers handle dtype

    # diffusers handles its own loading
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

    # ========== SGLang Optimization Settings ==========

    # VAE optimizations
    vae_tiling: bool = False  # Enable VAE tiling for high-resolution images
    vae_slicing: bool = False  # Enable VAE slicing for batch processing
    vae_sp: bool = False  # VAE sequence parallelism (not supported in diffusers)

    # Memory optimizations
    attention_slicing: bool = False  # Enable attention slicing for memory efficiency
    enable_model_cpu_offload: bool = False  # Offload model to CPU when not in use
    enable_sequential_cpu_offload: bool = False  # More aggressive CPU offloading

    # Attention backend replacement
    use_sglang_attention: bool = False  # Replace diffusers attention with SGLang backends
    attention_backend: str = "auto"  # "auto", "flash_attn", "sage_attn", "sage_attn_3", "sdpa"

    # Data parallelism (multi-GPU)
    enable_data_parallel: bool = False  # Enable simple data parallelism
    dp_devices: list[int] | None = None  # List of GPU device IDs to use

    def check_pipeline_config(self) -> None:
        """
        Override to skip most validation since diffusers handles its own config.
        Validate SGLang-specific optimization settings.
        """
        # Validate attention backend (accept CLI, diffusers, and SGLang names)
        valid_backends = {
            # Auto
            "auto",
            # SGLang CLI names
            "fa", "fa3", "fa4",
            "sage_attn", "sage_attn_three",
            "torch_sdpa",
            # Diffusers backend names
            "flash", "_flash_3", "_flash_3_hub",
            "sage", "sage_hub",
            "native", "xformers", "flex",
            # SGLang direct backends
            "sglang_fa", "sglang_sage", "sglang_sdpa",
            # Aliases
            "flash_attn", "sdpa",
        }
        if self.attention_backend not in valid_backends:
            raise ValueError(
                f"Invalid attention_backend '{self.attention_backend}'. "
                f"Must be one of: {sorted(valid_backends)}"
            )

        # Warn if conflicting CPU offload settings
        if self.enable_model_cpu_offload and self.enable_sequential_cpu_offload:
            import warnings

            warnings.warn(
                "Both enable_model_cpu_offload and enable_sequential_cpu_offload are True. "
                "Using enable_sequential_cpu_offload (more aggressive)."
            )

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

    @classmethod
    def with_optimizations(
        cls,
        vae_tiling: bool = True,
        attention_slicing: bool = True,
        use_sglang_attention: bool = True,
        attention_backend: str = "auto",
        **kwargs,
    ) -> "DiffusersGenericPipelineConfig":
        """
        Factory method to create a config with common optimizations enabled.

        Args:
            vae_tiling: Enable VAE tiling for high-resolution outputs
            attention_slicing: Enable attention slicing for memory efficiency
            use_sglang_attention: Use SGLang attention backends
            attention_backend: Which attention backend to use
            **kwargs: Additional config parameters

        Returns:
            DiffusersGenericPipelineConfig with optimizations enabled
        """
        return cls(
            vae_tiling=vae_tiling,
            attention_slicing=attention_slicing,
            use_sglang_attention=use_sglang_attention,
            attention_backend=attention_backend,
            **kwargs,
        )
