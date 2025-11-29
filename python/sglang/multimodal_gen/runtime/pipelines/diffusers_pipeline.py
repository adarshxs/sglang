# SPDX-License-Identifier: Apache-2.0
"""
Diffusers backend pipeline wrapper.

This module provides a wrapper that allows running any diffusers-supported model
through sglang's infrastructure using vanilla diffusers pipelines.
"""

import argparse
import re
import warnings
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as T
from diffusers import DiffusionPipeline
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DiffusersExecutionStage(PipelineStage):
    """Pipeline stage that wraps diffusers pipeline execution."""

    def __init__(self, diffusers_pipe: Any):
        super().__init__()
        self.diffusers_pipe = diffusers_pipe

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute the diffusers pipeline."""

        kwargs = self._build_pipeline_kwargs(batch, server_args)

        # Request tensor output for cleaner handling
        if "output_type" not in kwargs:
            kwargs["output_type"] = "pt"

        with torch.no_grad(), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                output = self.diffusers_pipe(**kwargs)
            except TypeError as e:
                # Some pipelines don't support output_type="pt"
                if "output_type" in str(e):
                    kwargs.pop("output_type", None)
                    output = self.diffusers_pipe(**kwargs)
                else:
                    raise

        batch.output = self._extract_output(output)
        if batch.output is not None:
            batch.output = self._postprocess_output(batch.output)

        return batch

    def _extract_output(self, output: Any) -> torch.Tensor | None:
        """Extract tensor output from pipeline result."""
        for attr in ["images", "frames", "video", "sample", "pred_original_sample"]:
            if not hasattr(output, attr):
                continue

            data = getattr(output, attr)
            if data is None:
                continue

            result = self._convert_to_tensor(data)
            if result is not None:
                logger.info(
                    "Extracted output from '%s': shape=%s, dtype=%s",
                    attr,
                    result.shape,
                    result.dtype,
                )
                return result

        logger.warning("Could not extract output from pipeline result")
        return None

    def _convert_to_tensor(self, data: Any) -> torch.Tensor | None:
        """Convert various data formats to a tensor."""
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            # (B, H, W, C) -> (B, C, H, W) or (B, T, H, W, C) -> (B, C, T, H, W)
            if tensor.ndim == 4:
                tensor = tensor.permute(0, 3, 1, 2)
            elif tensor.ndim == 5:
                tensor = tensor.permute(0, 4, 1, 2, 3)
            return tensor

        if hasattr(data, "mode"):  # PIL Image
            return T.ToTensor()(data)

        if isinstance(data, list) and len(data) > 0:
            return self._convert_list_to_tensor(data)

        return None

    def _convert_list_to_tensor(self, data: list) -> torch.Tensor | None:
        """Convert a list of items to a tensor."""
        first = data[0]

        # Nested list (e.g., [[frame1, frame2, ...]] for video batches)
        if isinstance(first, list) and len(first) > 0:
            data = first
            first = data[0]

        if hasattr(first, "mode"):  # PIL images
            tensors = [T.ToTensor()(img) for img in data]
            stacked = torch.stack(tensors)
            if len(tensors) > 1:
                return stacked.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
            return stacked[0]

        if isinstance(first, torch.Tensor):
            stacked = torch.stack(data)
            if len(data) > 1:
                return stacked.permute(1, 0, 2, 3)
            return stacked[0]

        if isinstance(first, np.ndarray):
            tensors = [torch.from_numpy(arr).float() for arr in data]
            if tensors[0].max() > 1.0:
                tensors = [t / 255.0 for t in tensors]
            if tensors[0].ndim == 3:
                tensors = [t.permute(2, 0, 1) for t in tensors]
            stacked = torch.stack(tensors)
            if len(data) > 1:
                return stacked.permute(1, 0, 2, 3)
            return stacked[0]

        return None

    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """Post-process output tensor to ensure valid values and correct shape."""
        output = output.cpu().float()

        # Handle NaN or Inf values
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning("Output contains invalid values, fixing...")
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)

        # Normalize to [0, 1] range if needed
        min_val, max_val = output.min().item(), output.max().item()
        if min_val < -0.5 or max_val > 1.5:
            output = (output + 1) / 2

        output = output.clamp(0, 1)

        # Ensure correct shape for downstream processing
        output = self._fix_output_shape(output)

        logger.info("Final output tensor shape: %s", output.shape)
        return output

    def _fix_output_shape(self, output: torch.Tensor) -> torch.Tensor:
        """Fix tensor shape for downstream processing.

        Expected: (B, C, H, W) for images or (B, C, T, H, W) for videos.
        """
        if output.dim() == 5:
            # Video: (B, T, C, H, W) -> (B, C, T, H, W)
            return output.permute(0, 2, 1, 3, 4)

        if output.dim() == 4:
            if output.shape[0] == 1 or output.shape[1] in [1, 3, 4]:
                return output  # Already (B, C, H, W)
            # (T, C, H, W) -> (1, C, T, H, W)
            return output.unsqueeze(0).permute(0, 2, 1, 3, 4)

        if output.dim() == 3:
            c, h, w = output.shape
            if c > 4 and w <= 4:
                output = output.permute(2, 0, 1)
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            return output.unsqueeze(0)

        if output.dim() == 2:
            return output.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        return output

    def _build_pipeline_kwargs(self, batch: Req, server_args: ServerArgs) -> dict:
        """Build kwargs dict for diffusers pipeline call."""
        kwargs = {}

        if batch.prompt is not None:
            kwargs["prompt"] = batch.prompt

        if batch.negative_prompt:
            kwargs["negative_prompt"] = batch.negative_prompt

        if batch.num_inference_steps is not None:
            kwargs["num_inference_steps"] = batch.num_inference_steps

        if batch.guidance_scale is not None:
            kwargs["guidance_scale"] = batch.guidance_scale

        if batch.height is not None:
            kwargs["height"] = batch.height

        if batch.width is not None:
            kwargs["width"] = batch.width

        if batch.num_frames is not None and batch.num_frames > 1:
            kwargs["num_frames"] = batch.num_frames

        # Generator for reproducibility
        if batch.generator is not None:
            kwargs["generator"] = batch.generator
        elif batch.seed is not None:
            device = self._get_pipeline_device()
            kwargs["generator"] = torch.Generator(device=device).manual_seed(batch.seed)

        # Image input for img2img or inpainting
        image = self._load_input_image(batch)
        if image is not None:
            kwargs["image"] = image

        if batch.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = batch.num_outputs_per_prompt

        # Extra diffusers-specific kwargs
        if batch.extra:
            diffusers_kwargs = batch.extra.get("diffusers_kwargs", {})
            if diffusers_kwargs:
                kwargs.update(diffusers_kwargs)

        return kwargs

    def _get_pipeline_device(self) -> str:
        """Get the device the pipeline is running on."""
        for attr in ["unet", "transformer", "vae"]:
            component = getattr(self.diffusers_pipe, attr, None)
            if component is not None:
                try:
                    return next(component.parameters()).device
                except StopIteration:
                    pass
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_input_image(self, batch: Req) -> Image.Image | None:
        """Load input image from batch."""
        # Check for PIL image in condition_image or pixel_values
        if batch.condition_image is not None and isinstance(
            batch.condition_image, Image.Image
        ):
            return batch.condition_image
        if batch.pixel_values is not None and isinstance(
            batch.pixel_values, Image.Image
        ):
            return batch.pixel_values

        if not batch.image_path:
            return None

        try:
            if batch.image_path.startswith(("http://", "https://")):
                response = requests.get(batch.image_path, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(batch.image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image from %s: %s", batch.image_path, e)
            return None


class DiffusersPipeline(ComposedPipelineBase):
    """
    Pipeline wrapper that uses vanilla diffusers pipelines.

    This allows running any diffusers-supported model through sglang's infrastructure
    without requiring native sglang implementation.
    """

    pipeline_name = "DiffusersPipeline"
    is_video_pipeline = False
    _required_config_modules: list[str] = []

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ):
        self.server_args = server_args
        self.model_path = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.modules: dict[str, Any] = {}
        self.post_init_called = False
        self.executor = executor or SyncExecutor(server_args=server_args)

        logger.info("Loading diffusers pipeline from %s", model_path)
        self.diffusers_pipe = self._load_diffusers_pipeline(model_path, server_args)
        self._detect_pipeline_type()

    def _load_diffusers_pipeline(self, model_path: str, server_args: ServerArgs) -> Any:
        """Load the diffusers pipeline."""

        original_model_path = model_path  # Keep original for custom_pipeline
        model_path = maybe_download_model(model_path)
        self.model_path = model_path

        dtype = self._get_dtype(server_args)
        logger.info("Loading diffusers pipeline with dtype=%s", dtype)

        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        except AttributeError as e:
            if "has no attribute" in str(e):
                # Custom pipeline class not in diffusers - try loading with custom_pipeline
                logger.info(
                    "Pipeline class not found in diffusers, trying custom_pipeline from repo..."
                )
                try:
                    pipe = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        custom_pipeline=original_model_path,
                        trust_remote_code=True,
                        revision=server_args.revision,
                    )
                except Exception as e2:
                    match = re.search(r"has no attribute (\w+)", str(e))
                    class_name = match.group(1) if match else "unknown"
                    raise RuntimeError(
                        f"Pipeline class '{class_name}' not found in diffusers and no custom pipeline.py in repo. "
                        f"Try: pip install --upgrade diffusers (some pipelines require latest version). "
                        f"Original error: {e}"
                    ) from e2
            else:
                raise
        except Exception as e:
            # Only retry with float32 for dtype-related errors
            if "dtype" in str(e).lower() or "float" in str(e).lower():
                logger.warning(
                    "Failed with dtype=%s, falling back to float32: %s", dtype, e
                )
                pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
            else:
                raise

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipe = pipe.to("mps")

        logger.info("Loaded diffusers pipeline: %s", pipe.__class__.__name__)

        # Apply SGLang optimizations
        pipe = self._apply_optimizations(pipe, server_args)

        return pipe

    def _apply_optimizations(self, pipe: Any, server_args: ServerArgs) -> Any:
        """Apply SGLang optimizations to the diffusers pipeline.

        Optimizations applied:
        1. torch.compile for transformer/unet (if enabled)
        2. VAE tiling for high-resolution outputs
        3. VAE slicing for batch processing
        4. Attention slicing for memory efficiency
        5. CPU offloading for memory-constrained systems
        """
        pipeline_config = getattr(server_args, "pipeline_config", None)

        # 1. torch.compile optimization
        if server_args.enable_torch_compile:
            pipe = self._apply_torch_compile(pipe)

        # 2. VAE optimizations
        if pipeline_config and getattr(pipeline_config, "vae_tiling", False):
            self._enable_vae_tiling(pipe)

        if pipeline_config and getattr(pipeline_config, "vae_slicing", False):
            self._enable_vae_slicing(pipe)

        # 3. Attention slicing for memory efficiency
        if pipeline_config and getattr(pipeline_config, "attention_slicing", False):
            self._enable_attention_slicing(pipe)

        # 4. CPU offloading
        if pipeline_config and getattr(pipeline_config, "enable_model_cpu_offload", False):
            self._enable_cpu_offload(pipe, sequential=False)
        elif pipeline_config and getattr(
            pipeline_config, "enable_sequential_cpu_offload", False
        ):
            self._enable_cpu_offload(pipe, sequential=True)

        # 5. SGLang attention backend replacement
        self._apply_attention_backend(pipe, server_args)

        # 6. Data parallelism
        if pipeline_config and getattr(pipeline_config, "enable_data_parallel", False):
            pipe = self._apply_data_parallel(
                pipe, getattr(pipeline_config, "dp_devices", None)
            )

        return pipe

    def _apply_data_parallel(
        self, pipe: Any, device_ids: list[int] | None = None
    ) -> Any:
        """Apply data parallelism for multi-GPU inference.

        Args:
            pipe: The diffusers pipeline
            device_ids: List of GPU device IDs to use

        Returns:
            Pipeline with DataParallel-wrapped models
        """
        try:
            from sglang.multimodal_gen.runtime.optimizations.data_parallel import (
                apply_data_parallel,
            )

            pipe = apply_data_parallel(pipe, device_ids=device_ids)
        except Exception as e:
            logger.warning("Failed to apply data parallelism: %s", e)

        return pipe

    def _apply_torch_compile(self, pipe: Any) -> Any:
        """Apply torch.compile to transformer/unet components."""
        compiled_components = []

        # Compile transformer (for DiT-based models like FLUX, SD3, etc.)
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            try:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="max-autotune", fullgraph=False
                )
                compiled_components.append("transformer")
            except Exception as e:
                logger.warning("Failed to compile transformer: %s", e)

        # Compile unet (for UNet-based models like SD 1.5, SDXL, etc.)
        if hasattr(pipe, "unet") and pipe.unet is not None:
            try:
                pipe.unet = torch.compile(
                    pipe.unet, mode="max-autotune", fullgraph=False
                )
                compiled_components.append("unet")
            except Exception as e:
                logger.warning("Failed to compile unet: %s", e)

        if compiled_components:
            logger.info(
                "torch.compile enabled for: %s", ", ".join(compiled_components)
            )
        else:
            logger.warning(
                "torch.compile requested but no compilable components found"
            )

        return pipe

    def _enable_vae_tiling(self, pipe: Any) -> None:
        """Enable VAE tiling for high-resolution image generation."""
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                logger.info("VAE tiling enabled")
            elif hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
                logger.info("VAE tiling enabled (pipeline method)")

    def _enable_vae_slicing(self, pipe: Any) -> None:
        """Enable VAE slicing for batch processing efficiency."""
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                logger.info("VAE slicing enabled")
            elif hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
                logger.info("VAE slicing enabled (pipeline method)")

    def _enable_attention_slicing(self, pipe: Any) -> None:
        """Enable attention slicing for memory efficiency."""
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("auto")
            logger.info("Attention slicing enabled")

    def _enable_cpu_offload(self, pipe: Any, sequential: bool = False) -> None:
        """Enable CPU offloading for memory-constrained systems.

        Args:
            pipe: The diffusers pipeline
            sequential: If True, use sequential CPU offload (more aggressive memory saving)
        """
        try:
            if sequential:
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                    logger.info("Sequential CPU offload enabled")
            else:
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    logger.info("Model CPU offload enabled")
        except Exception as e:
            logger.warning("Failed to enable CPU offload: %s", e)

    def _apply_attention_backend(self, pipe: Any, server_args: ServerArgs) -> None:
        """Apply SGLang attention backend to the diffusers pipeline.

        This replaces diffusers' default attention with SGLang's optimized backends.

        Args:
            pipe: The diffusers pipeline
            server_args: Server arguments containing attention backend config
        """
        pipeline_config = getattr(server_args, "pipeline_config", None)
        if not pipeline_config:
            return

        # Check if attention backend replacement is enabled
        if not getattr(pipeline_config, "use_sglang_attention", False):
            return

        # Get backend from server_args or pipeline_config
        backend = server_args.attention_backend
        if backend is None:
            backend = getattr(pipeline_config, "attention_backend", "auto")

        try:
            from sglang.multimodal_gen.runtime.optimizations.diffusers_attention import (
                apply_sglang_attention,
            )

            apply_sglang_attention(pipe, backend=backend)
        except Exception as e:
            logger.warning("Failed to apply SGLang attention backend: %s", e)

    def _get_dtype(self, server_args: ServerArgs) -> torch.dtype:
        """Determine the dtype to use for model loading."""
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if hasattr(server_args, "pipeline_config") and server_args.pipeline_config:
            dit_precision = getattr(server_args.pipeline_config, "dit_precision", None)
            if dit_precision == "fp16":
                dtype = torch.float16
            elif dit_precision == "bf16":
                dtype = torch.bfloat16
            elif dit_precision == "fp32":
                dtype = torch.float32

        return dtype

    def _detect_pipeline_type(self):
        """Detect if this is an image or video pipeline."""
        pipe_class_name = self.diffusers_pipe.__class__.__name__.lower()
        video_indicators = ["video", "animat", "cogvideo", "wan", "hunyuan"]
        self.is_video_pipeline = any(ind in pipe_class_name for ind in video_indicators)
        logger.info(
            "Detected pipeline type: %s",
            "video" if self.is_video_pipeline else "image",
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Skip sglang's module loading - diffusers handles it."""
        return {"diffusers_pipeline": self.diffusers_pipe}

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Create the execution stage wrapping the diffusers pipeline."""
        self.add_stage(
            stage_name="diffusers_execution",
            stage=DiffusersExecutionStage(self.diffusers_pipe),
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        """Initialize the pipeline."""
        pass

    def post_init(self) -> None:
        """Post initialization hook."""
        if self.post_init_called:
            return
        self.post_init_called = True
        self.initialize_pipeline(self.server_args)
        self.create_pipeline_stages(self.server_args)

    def add_stage(self, stage_name: str, stage: PipelineStage):
        """Add a stage to the pipeline."""
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    @property
    def stages(self) -> list[PipelineStage]:
        """List of stages in the pipeline."""
        return self._stages

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute the pipeline on the given batch."""
        if not self.post_init_called:
            self.post_init()
        return self.executor.execute(self.stages, batch, server_args)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        pipeline_config: str | PipelineConfig | None = None,
        args: argparse.Namespace | None = None,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        **kwargs,
    ) -> "DiffusersPipeline":
        """Load a pipeline from a pretrained model using diffusers backend."""
        kwargs["model_path"] = model_path
        server_args = ServerArgs.from_kwargs(**kwargs)

        pipe = cls(
            model_path,
            server_args,
            required_config_modules=required_config_modules,
            loaded_modules=loaded_modules,
        )
        pipe.post_init()
        return pipe

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        """Get a module by name."""
        if module_name == "diffusers_pipeline":
            return self.diffusers_pipe
        return self.modules.get(module_name, default_value)


EntryClass = DiffusersPipeline
