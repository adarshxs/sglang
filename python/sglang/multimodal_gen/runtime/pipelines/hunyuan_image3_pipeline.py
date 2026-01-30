from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_image3 import (
    HunyuanImage3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    HunyuanImage3SamplingParams,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class HunyuanImage3SizeParser:
    """
    Parse a size string like "{height}x{width}" and align to multiples of 16.
    """

    size_str: str

    def __post_init__(self) -> None:
        try:
            h_str, w_str = self.size_str.strip().lower().split("x")
            self.original_height = int(h_str)
            self.original_width = int(w_str)
        except Exception as exc:
            raise ValueError(
                f'Invalid size string "{self.size_str}". Expected "HxW", e.g., 1024x1024.'
            ) from exc

        self.aligned_height = self._ceil16(self.original_height)
        self.aligned_width = self._ceil16(self.original_width)

    def is_valid(self) -> bool:
        return 512 <= self.original_width <= 2048 and 512 <= self.original_height <= 2048

    def get_aligned_size(self) -> tuple[int, int]:
        return self.aligned_height, self.aligned_width

    @staticmethod
    def _ceil16(x: int) -> int:
        return (x + 15) // 16 * 16


class HunyuanImage3GenerateStage(PipelineStage):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def _resolve_image_size(self, batch: Req, explicit_size: str | None = None) -> str:
        explicit_size = explicit_size or getattr(batch, "image_size", None)
        if explicit_size:
            size_str = explicit_size
        elif batch.height is not None and batch.width is not None:
            size_str = f"{batch.height}x{batch.width}"
        else:
            base_size = getattr(getattr(self.model, "config", None), "image_base_size", 1024)
            size_str = f"{base_size}x{base_size}"

        size_parser = HunyuanImage3SizeParser(size_str)
        if not size_parser.is_valid():
            raise ValueError(
                f"Unsupported image size {size_str}. "
                "Height/width must be within [512, 2048]."
            )
        aligned_h, aligned_w = size_parser.get_aligned_size()
        aligned_size = f"{aligned_h}x{aligned_w}"

        image_processor = getattr(self.model, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "build_image_info"):
            try:
                image_info = image_processor.build_image_info(aligned_size)
                return f"{image_info.h}x{image_info.w}"
            except Exception:
                return aligned_size

        return aligned_size

    def _normalize_images(self, output: Any) -> list[np.ndarray]:
        if isinstance(output, tuple) and len(output) == 2:
            output = output[1]

        if isinstance(output, list):
            images = output
        else:
            images = [output]

        normalized: list[np.ndarray] = []
        for img in images:
            if isinstance(img, np.ndarray):
                normalized.append(img)
            elif hasattr(img, "mode"):
                normalized.append(np.asarray(img))
            elif isinstance(img, torch.Tensor):
                normalized.append(img.detach().cpu().float().numpy())
            else:
                raise TypeError(f"Unsupported image output type: {type(img)}")
        return normalized

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if self.model is None:
            raise RuntimeError("HunyuanImage3 model is not loaded on this rank.")

        if server_args.num_gpus > 1:
            logger.warning(
                "HunyuanImage3 pipeline runs image generation on rank 0 only."
            )

        prompt = batch.prompt
        if prompt is None:
            raise ValueError("HunyuanImage3 requires a text prompt.")
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError(
                    "HunyuanImage3 pipeline supports a single prompt per request."
                )
            prompt = prompt[0]

        bot_task = getattr(batch, "bot_task", None) or "image"
        task_extra_kwargs = dict(getattr(batch, "task_extra_kwargs", None) or {})
        explicit_size = task_extra_kwargs.pop("image_size", None)
        image_size = self._resolve_image_size(batch, explicit_size)
        if (
            batch.num_inference_steps is not None
            and "diff_infer_steps" not in task_extra_kwargs
        ):
            task_extra_kwargs["diff_infer_steps"] = batch.num_inference_steps
        seeds = batch.seeds or [batch.seed or 0]

        outputs: list[np.ndarray] = []
        for seed in seeds:
            result = self.model.generate_image(
                prompt=prompt,
                seed=seed,
                image_size=image_size,
                bot_task=bot_task,
                output_type="np",
                **task_extra_kwargs,
            )
            outputs.extend(self._normalize_images(result))

        return OutputBatch(output=outputs, timings=batch.timings)


class HunyuanImage3Pipeline(ComposedPipelineBase):
    pipeline_name = "HunyuanImage3Pipeline"
    pipeline_config_cls = HunyuanImage3PipelineConfig
    sampling_params_cls = HunyuanImage3SamplingParams

    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if get_world_rank() != 0:
            return {}

        try:
            from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
        except Exception as exc:
            raise RuntimeError(
                "hunyuan_image_3 is required for HunyuanImage3Pipeline. "
                "Install it from the HunyuanImage-3.0 repository "
                "(pip install git+https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git)."
            ) from exc

        model = HunyuanImage3ForCausalMM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        model = model.to(get_local_torch_device())
        model.eval()
        if hasattr(model, "load_tokenizer"):
            model.load_tokenizer(self.model_path)
        return {"hunyuan_image3_model": model}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="hunyuan_image3_generate_stage",
            stage=HunyuanImage3GenerateStage(
                model=self.get_module("hunyuan_image3_model")
            ),
        )


class HunyuanImage3PipelineAlias(HunyuanImage3Pipeline):
    pipeline_name = "HunyuanImage3"


EntryClass = [HunyuanImage3Pipeline, HunyuanImage3PipelineAlias]
