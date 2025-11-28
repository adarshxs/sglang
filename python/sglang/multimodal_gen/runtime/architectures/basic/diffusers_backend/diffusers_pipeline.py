# SPDX-License-Identifier: Apache-2.0
"""
Diffusers backend pipeline wrapper.

This module provides a wrapper that allows running any diffusers-supported model
through sglang's infrastructure using vanilla diffusers pipelines.
"""

import argparse
from typing import Any

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
        # Build kwargs from batch
        kwargs = self._build_pipeline_kwargs(batch, server_args)

        logger.info(
            "Executing diffusers pipeline with kwargs: %s",
            {k: type(v).__name__ for k, v in kwargs.items()},
        )

        # Execute the diffusers pipeline
        with torch.no_grad():
            output = self.diffusers_pipe(**kwargs)

        # Extract outputs - diffusers pipelines return different output types
        if hasattr(output, "images"):
            # Image pipelines (StableDiffusion, FLUX, etc.)
            images = output.images
            if isinstance(images, list):
                # Convert PIL images to tensors if needed
                if hasattr(images[0], "mode"):  # PIL Image
                    import torchvision.transforms as T

                    transform = T.ToTensor()
                    tensors = [transform(img) for img in images]
                    batch.output = torch.stack(tensors) if len(tensors) > 1 else tensors[0]
                else:
                    batch.output = torch.stack(images) if len(images) > 1 else images[0]
            else:
                batch.output = images
        elif hasattr(output, "frames"):
            # Video pipelines
            frames = output.frames
            if isinstance(frames, list):
                # Handle list of frame tensors or PIL images
                if len(frames) > 0:
                    if hasattr(frames[0], "mode"):  # PIL Image
                        import torchvision.transforms as T
                        transform = T.ToTensor()
                        tensors = [transform(img) for img in frames]
                        batch.output = torch.stack(tensors)
                    else:
                        batch.output = torch.stack(frames) if len(frames) > 1 else frames[0]
                else:
                    batch.output = None
            else:
                batch.output = frames
        else:
            # Fallback - try to get the first attribute that looks like output
            for attr in ["sample", "pred_original_sample", "latents"]:
                if hasattr(output, attr):
                    batch.output = getattr(output, attr)
                    break
            else:
                # Last resort - assume output is the result directly
                batch.output = output

        # Ensure output tensor is in valid range and handle NaN/Inf
        if batch.output is not None and isinstance(batch.output, torch.Tensor):
            # Check for NaN or Inf values
            if torch.isnan(batch.output).any() or torch.isinf(batch.output).any():
                logger.warning(
                    "Output contains NaN or Inf values. Clamping to valid range."
                )
                batch.output = torch.nan_to_num(batch.output, nan=0.0, posinf=1.0, neginf=0.0)

            # Ensure values are in [0, 1] range for image outputs
            if batch.output.min() < 0 or batch.output.max() > 1:
                # Some models output in [-1, 1] range, normalize to [0, 1]
                if batch.output.min() >= -1 and batch.output.max() <= 1:
                    batch.output = (batch.output + 1) / 2
                else:
                    # Clamp to valid range
                    batch.output = batch.output.clamp(0, 1)

        return batch

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

        if batch.guidance_scale is not None and batch.guidance_scale > 0:
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

        # Number of outputs
        if batch.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = batch.num_outputs_per_prompt

        # Request PIL output for easier handling
        kwargs["output_type"] = "pil"

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

