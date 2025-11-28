# SPDX-License-Identifier: Apache-2.0
"""
Diffusers backend pipeline wrapper.

This module provides a wrapper that allows running any diffusers-supported model
through sglang's infrastructure using vanilla diffusers pipelines.
"""

import argparse
from typing import Any
from PIL import Image

import torch

from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DiffusersExecutionStage(PipelineStage):
    """
    A single stage that wraps the entire diffusers pipeline execution.
    """

    def __init__(self, diffusers_pipe: Any):
        super().__init__()
        self.diffusers_pipe = diffusers_pipe

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute the diffusers pipeline."""
        import warnings

        # Build kwargs from batch
        kwargs = self._build_pipeline_kwargs(batch, server_args)

        logger.info(
            "Executing diffusers pipeline with kwargs: %s",
            {k: type(v).__name__ for k, v in kwargs.items()},
        )

        # Execute the diffusers pipeline, catching warnings
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            output = self.diffusers_pipe(**kwargs)

            # Check if there were NaN warnings during generation
            nan_warning = any("invalid value" in str(w.message) for w in caught_warnings)
            if nan_warning:
                logger.warning("NaN values detected during pipeline execution")

        # Extract output
        batch.output = self._extract_output(output)

        # Post-process output
        if batch.output is not None:
            batch.output = self._postprocess_output(batch.output)

        return batch

    def _extract_output(self, output: Any) -> torch.Tensor | None:
        """Extract tensor output from pipeline result."""
        import torchvision.transforms as T

        result = None

        logger.debug("Pipeline output type: %s", type(output).__name__)
        logger.debug("Pipeline output attributes: %s", [a for a in dir(output) if not a.startswith("_")])

        # Try to get images
        if hasattr(output, "images"):
            images = output.images
            logger.debug("Found images attribute, type: %s", type(images).__name__)

            if isinstance(images, list) and len(images) > 0:
                logger.debug("Images is list with %d items, first item type: %s",
                           len(images), type(images[0]).__name__)

                if hasattr(images[0], "mode"):  # PIL Image
                    logger.debug("Converting PIL images to tensors, size: %s", images[0].size)
                    transform = T.ToTensor()
                    tensors = [transform(img) for img in images]
                    result = torch.stack(tensors) if len(tensors) > 1 else tensors[0]
                elif isinstance(images[0], torch.Tensor):
                    logger.debug("Images are tensors, shape: %s", images[0].shape)
                    result = torch.stack(images) if len(images) > 1 else images[0]
            elif isinstance(images, torch.Tensor):
                logger.debug("Images is tensor, shape: %s", images.shape)
                result = images
            elif hasattr(images, "mode"):  # Single PIL image
                logger.debug("Single PIL image, size: %s", images.size)
                result = T.ToTensor()(images)

        # Try to get frames (video)
        elif hasattr(output, "frames"):
            frames = output.frames
            logger.debug("Found frames attribute, type: %s", type(frames).__name__)

            if isinstance(frames, list) and len(frames) > 0:
                if hasattr(frames[0], "mode"):  # PIL Image
                    transform = T.ToTensor()
                    tensors = [transform(img) for img in frames]
                    result = torch.stack(tensors)
                elif isinstance(frames[0], torch.Tensor):
                    result = torch.stack(frames) if len(frames) > 1 else frames[0]
            elif isinstance(frames, torch.Tensor):
                result = frames

        # Fallback to other attributes
        if result is None:
            for attr in ["sample", "pred_original_sample"]:
                if hasattr(output, attr):
                    val = getattr(output, attr)
                    if isinstance(val, torch.Tensor):
                        logger.debug("Using fallback attribute '%s', shape: %s", attr, val.shape)
                        result = val
                        break

        if result is not None:
            logger.info("Extracted output tensor shape: %s, dtype: %s", result.shape, result.dtype)
        else:
            logger.warning("Could not extract output from pipeline result")

        return result

    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """Post-process output tensor to ensure valid values and correct shape."""
        # Handle NaN or Inf values
        has_invalid = torch.isnan(output).any() or torch.isinf(output).any()
        if has_invalid:
            logger.warning("Output contains NaN or Inf values. Fixing...")

            # Count invalid pixels
            nan_count = torch.isnan(output).sum().item()
            inf_count = torch.isinf(output).sum().item()
            total = output.numel()
            logger.warning(
                "Invalid values: %d NaN (%.1f%%), %d Inf (%.1f%%)",
                nan_count, 100 * nan_count / total,
                inf_count, 100 * inf_count / total,
            )

            # Replace with neutral gray value
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)

        # Normalize to [0, 1] range
        min_val, max_val = output.min().item(), output.max().item()

        if min_val < -0.5 or max_val > 1.5:
            # Likely in [-1, 1] range, normalize
            output = (output + 1) / 2

        # Clamp to valid range
        output = output.clamp(0, 1)

        # Ensure correct tensor shape for downstream processing
        # Expected format: (C, H, W) for images or (C, T, H, W) for videos
        # The downstream post_process_sample expects at least 3D
        if output.dim() == 2:
            # (H, W) -> (1, H, W) - grayscale image
            output = output.unsqueeze(0)
        elif output.dim() == 4:
            # (B, C, H, W) -> (C, H, W) if batch size is 1
            if output.shape[0] == 1:
                output = output.squeeze(0)

        # For images, ensure we have (C, H, W) format with C=3 for RGB
        if output.dim() == 3:
            c, h, w = output.shape
            # If channels are last (H, W, C), transpose
            if c > 4 and w <= 4:
                output = output.permute(2, 0, 1)
                c, h, w = output.shape
            # If grayscale, expand to RGB
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)  # Use repeat instead of expand for contiguous memory

        # The downstream code expects output to be a list or iterable of samples
        # where each sample is (C, H, W) for images or (C, T, H, W) for videos.
        # When iterating over output_batch.output, it should yield complete samples.
        # For a single image, we need to wrap it in a list-like structure.
        # We'll add a batch dimension so output[0] gives the full image.
        if output.dim() == 3:
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            # Then when code does `for sample in output`, sample will be (C, H, W)
            output = output.unsqueeze(0)

        logger.info("Final output tensor shape: %s", output.shape)
        return output

    def _build_pipeline_kwargs(self, batch: Req, server_args: ServerArgs) -> dict:
        """Build kwargs dict for diffusers pipeline call."""
        kwargs = {}

        # Required params
        if batch.prompt is not None:
            kwargs["prompt"] = batch.prompt

        if batch.negative_prompt is not None and batch.negative_prompt:
            kwargs["negative_prompt"] = batch.negative_prompt

        # Generation params
        if batch.num_inference_steps is not None:
            kwargs["num_inference_steps"] = batch.num_inference_steps

        # Always pass guidance_scale (some models need 0.0 explicitly)
        if batch.guidance_scale is not None:
            kwargs["guidance_scale"] = batch.guidance_scale

        # Dimensions
        if batch.height is not None:
            kwargs["height"] = batch.height

        if batch.width is not None:
            kwargs["width"] = batch.width

        # Video-specific params
        if batch.num_frames is not None and batch.num_frames > 1:
            kwargs["num_frames"] = batch.num_frames

        # Generator for reproducibility
        if batch.generator is not None:
            kwargs["generator"] = batch.generator
        elif batch.seed is not None:
            # Determine device for generator
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(self.diffusers_pipe, "unet") and self.diffusers_pipe.unet is not None:
                try:
                    device = next(self.diffusers_pipe.unet.parameters()).device
                except StopIteration:
                    pass
            elif hasattr(self.diffusers_pipe, "transformer") and self.diffusers_pipe.transformer is not None:
                try:
                    device = next(self.diffusers_pipe.transformer.parameters()).device
                except StopIteration:
                    pass
            kwargs["generator"] = torch.Generator(device=device).manual_seed(batch.seed)

        # Image input for img2img or inpainting
        if batch.pil_image is not None:
            kwargs["image"] = batch.pil_image
        elif batch.image_path is not None and batch.image_path:
            try:
                image = Image.open(batch.image_path).convert("RGB")
                kwargs["image"] = image
                logger.info("Loaded input image from %s, size: %s", batch.image_path, image.size)
            except Exception as e:
                logger.error("Failed to load image from %s: %s", batch.image_path, e)

        # Number of outputs
        if batch.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = batch.num_outputs_per_prompt

        # Add any extra diffusers-specific kwargs from batch.extra
        # This allows passing arbitrary parameters to the diffusers pipeline
        if batch.extra:
            diffusers_kwargs = batch.extra.get("diffusers_kwargs", {})
            if diffusers_kwargs:
                logger.info("Adding diffusers_kwargs: %s", diffusers_kwargs)
                kwargs.update(diffusers_kwargs)

        return kwargs


class DiffusersPipeline(ComposedPipelineBase):
    """
    A pipeline wrapper that uses vanilla diffusers pipelines.

    This allows running any diffusers-supported model through sglang's infrastructure
    without requiring native sglang implementation.
    """

    pipeline_name = "DiffusersPipeline"
    is_video_pipeline = False  # Will be set based on loaded pipeline

    # Empty since we don't use sglang's component loading
    _required_config_modules: list[str] = []

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ):
        """
        Initialize the DiffusersPipeline wrapper.

        Args:
            model_path: Path to the model or HuggingFace model ID
            server_args: Server arguments
            required_config_modules: Ignored for diffusers backend
            loaded_modules: Ignored for diffusers backend
            executor: Optional custom executor
        """
        # Don't call super().__init__ as we handle loading differently
        self.server_args = server_args
        self.model_path = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.modules: dict[str, Any] = {}
        self.post_init_called = False

        # Use sync executor by default for diffusers (simpler, works with all pipelines)
        self.executor = executor or SyncExecutor(server_args=server_args)

        # Load the diffusers pipeline
        logger.info("Loading diffusers pipeline from %s", model_path)
        self.diffusers_pipe = self._load_diffusers_pipeline(model_path, server_args)

        # Detect if this is a video pipeline
        self._detect_pipeline_type()

    def _load_diffusers_pipeline(
        self, model_path: str, server_args: ServerArgs
    ) -> Any:
        """Load the diffusers pipeline."""
        from diffusers import DiffusionPipeline

        # Download model if needed
        model_path = maybe_download_model(model_path)
        self.model_path = model_path

        # Determine dtype - try bfloat16 first for better compatibility with newer models
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if hasattr(server_args, "pipeline_config") and server_args.pipeline_config:
            dit_precision = getattr(
                server_args.pipeline_config, "dit_precision", None
            )
            if dit_precision == "fp16":
                dtype = torch.float16
            elif dit_precision == "bf16":
                dtype = torch.bfloat16
            elif dit_precision == "fp32":
                dtype = torch.float32

        # Load the pipeline
        logger.info("Loading diffusers pipeline with dtype=%s", dtype)
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        except Exception as e:
            # Fallback to float32 if there's an issue with the dtype
            logger.warning(
                "Failed to load with dtype=%s, falling back to float32: %s", dtype, e
            )
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        # Move to appropriate device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipe = pipe.to("mps")

        logger.info(
            "Loaded diffusers pipeline: %s", pipe.__class__.__name__
        )

        return pipe

    def _detect_pipeline_type(self):
        """Detect if this is an image or video pipeline."""
        pipe_class_name = self.diffusers_pipe.__class__.__name__.lower()
        video_indicators = ["video", "animat", "cogvideo", "wan", "hunyuan"]

        self.is_video_pipeline = any(
            indicator in pipe_class_name for indicator in video_indicators
        )
        logger.info(
            "Detected pipeline type: %s",
            "video" if self.is_video_pipeline else "image",
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Override to skip sglang's module loading.
        The diffusers pipeline handles its own module loading.
        """
        # Store reference to diffusers components
        return {"diffusers_pipeline": self.diffusers_pipe}

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Create the execution stage wrapping the diffusers pipeline."""
        self.add_stage(
            stage_name="diffusers_execution",
            stage=DiffusersExecutionStage(self.diffusers_pipe),
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        """Initialize the pipeline (called during post_init)."""
        # No additional initialization needed for diffusers backend
        pass

    def post_init(self) -> None:
        """Post initialization hook."""
        if self.post_init_called:
            return
        self.post_init_called = True

        self.initialize_pipeline(self.server_args)
        logger.info("Creating diffusers execution stage...")
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
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the pipeline on the given batch.

        Args:
            batch: The request batch to process
            server_args: Server arguments

        Returns:
            The batch with generated output
        """
        if not self.post_init_called:
            self.post_init()

        logger.info(
            "Running diffusers pipeline stages: %s",
            list(self._stage_name_mapping.keys()),
        )
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
        """
        Load a pipeline from a pretrained model using diffusers backend.

        Args:
            model_path: Path to the model or HuggingFace model ID
            device: Device to load the model on (optional)
            torch_dtype: Data type for model weights (optional)
            pipeline_config: Pipeline configuration (optional)
            args: Additional arguments (optional)
            required_config_modules: Ignored for diffusers backend
            loaded_modules: Ignored for diffusers backend
            **kwargs: Additional keyword arguments

        Returns:
            The loaded DiffusersPipeline
        """
        kwargs["model_path"] = model_path
        server_args = ServerArgs.from_kwargs(**kwargs)

        logger.info("Creating DiffusersPipeline with server_args: %s", server_args)

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


# Entry class for pipeline discovery
EntryClass = DiffusersPipeline

