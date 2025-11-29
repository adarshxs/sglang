# SPDX-License-Identifier: Apache-2.0
"""
Data parallelism utilities for diffusers pipelines.

This module provides simple data parallelism wrappers that enable multi-GPU inference
for diffusers pipelines without requiring complex distributed setup.
"""

from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def get_available_gpus() -> list[int]:
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


class DataParallelWrapper:
    """
    Simple data parallelism wrapper for diffusers pipelines.

    This wrapper enables batch processing across multiple GPUs by splitting
    the batch dimension across devices.

    Note: This is a simple implementation that works well for batch inference.
    For more advanced parallelism (tensor/sequence parallel), use SGLang's native
    pipelines which have deep integration with distributed primitives.
    """

    def __init__(
        self,
        pipe: Any,
        device_ids: list[int] | None = None,
    ):
        """
        Initialize data parallel wrapper.

        Args:
            pipe: The diffusers pipeline to wrap
            device_ids: List of GPU device IDs to use. If None, uses all available.
        """
        self.pipe = pipe
        self.device_ids = device_ids or get_available_gpus()

        if len(self.device_ids) < 2:
            logger.warning(
                "DataParallelWrapper initialized with less than 2 GPUs, "
                "no parallelism will be applied"
            )

        self._wrap_models()

    def _wrap_models(self) -> None:
        """Wrap transformer/unet models with DataParallel."""
        if len(self.device_ids) < 2:
            return

        wrapped = []

        # Wrap transformer (DiT models)
        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            if not isinstance(self.pipe.transformer, torch.nn.DataParallel):
                self.pipe.transformer = torch.nn.DataParallel(
                    self.pipe.transformer,
                    device_ids=self.device_ids,
                )
                wrapped.append("transformer")

        # Wrap unet (UNet models)
        if hasattr(self.pipe, "unet") and self.pipe.unet is not None:
            if not isinstance(self.pipe.unet, torch.nn.DataParallel):
                self.pipe.unet = torch.nn.DataParallel(
                    self.pipe.unet,
                    device_ids=self.device_ids,
                )
                wrapped.append("unet")

        if wrapped:
            logger.info(
                "DataParallel enabled for %s on GPUs %s",
                ", ".join(wrapped),
                self.device_ids,
            )

    def unwrap(self) -> Any:
        """
        Unwrap DataParallel modules and return the original pipeline.

        Returns:
            The original pipeline with unwrapped modules.
        """
        if hasattr(self.pipe, "transformer") and isinstance(
            self.pipe.transformer, torch.nn.DataParallel
        ):
            self.pipe.transformer = self.pipe.transformer.module

        if hasattr(self.pipe, "unet") and isinstance(
            self.pipe.unet, torch.nn.DataParallel
        ):
            self.pipe.unet = self.pipe.unet.module

        return self.pipe

    def __call__(self, *args, **kwargs):
        """Forward call to the wrapped pipeline."""
        return self.pipe(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped pipeline."""
        return getattr(self.pipe, name)


def apply_data_parallel(
    pipe: Any,
    device_ids: list[int] | None = None,
) -> Any:
    """
    Apply simple data parallelism to a diffusers pipeline.

    This wraps the transformer/unet with torch.nn.DataParallel for multi-GPU
    batch processing.

    Args:
        pipe: A diffusers pipeline
        device_ids: List of GPU device IDs to use. If None, uses all available.

    Returns:
        The pipeline with DataParallel-wrapped models.

    Example:
        ```python
        from diffusers import StableDiffusionPipeline
        from sglang.multimodal_gen.runtime.optimizations import apply_data_parallel

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to("cuda")
        pipe = apply_data_parallel(pipe, device_ids=[0, 1])
        ```
    """
    wrapper = DataParallelWrapper(pipe, device_ids)
    return wrapper.pipe


def enable_device_map_parallel(
    model_path: str,
    device_map: str | dict = "auto",
    **kwargs,
) -> Any:
    """
    Load a diffusers pipeline with automatic device mapping for multi-GPU.

    This uses Hugging Face's device_map feature to automatically distribute
    model layers across available GPUs.

    Args:
        model_path: Path or HuggingFace ID of the model
        device_map: Device map strategy. Options:
            - "auto": Automatically distribute across available GPUs
            - "balanced": Evenly distribute layers
            - dict: Custom mapping of layers to devices
        **kwargs: Additional arguments to pass to from_pretrained

    Returns:
        A diffusers pipeline with layers distributed across GPUs.

    Example:
        ```python
        from sglang.multimodal_gen.runtime.optimizations import enable_device_map_parallel

        pipe = enable_device_map_parallel(
            "black-forest-labs/FLUX.1-dev",
            device_map="auto",
        )
        ```
    """
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        device_map=device_map,
        **kwargs,
    )

    logger.info("Loaded pipeline with device_map='%s'", device_map)

    return pipe

