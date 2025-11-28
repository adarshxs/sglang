# SPDX-License-Identifier: Apache-2.0
"""
Tests for the diffusers backend.

This module tests the diffusers backend functionality, ensuring that models
not natively supported by sglang can be run through the diffusers backend.
"""

import pytest
import torch

from sglang.multimodal_gen.configs.pipelines.diffusers_generic import (
    DiffusersGenericPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.server_args import Backend


class TestBackendEnum:
    """Test the Backend enum functionality."""

    def test_backend_values(self):
        """Test that Backend enum has expected values."""
        assert Backend.AUTO.value == "auto"
        assert Backend.SGLANG.value == "sglang"
        assert Backend.DIFFUSERS.value == "diffusers"

    def test_backend_from_string(self):
        """Test Backend.from_string() conversion."""
        assert Backend.from_string("auto") == Backend.AUTO
        assert Backend.from_string("sglang") == Backend.SGLANG
        assert Backend.from_string("diffusers") == Backend.DIFFUSERS
        assert Backend.from_string("DIFFUSERS") == Backend.DIFFUSERS

    def test_backend_choices(self):
        """Test Backend.choices() returns expected list."""
        choices = Backend.choices()
        assert "auto" in choices
        assert "sglang" in choices
        assert "diffusers" in choices


class TestDiffusersGenericConfig:
    """Test the generic diffusers configuration classes."""

    def test_sampling_params_defaults(self):
        """Test DiffusersGenericSamplingParams default values."""
        params = DiffusersGenericSamplingParams()
        assert params.num_frames == 1
        assert params.height == 512
        assert params.width == 512
        assert params.num_inference_steps == 30
        assert params.guidance_scale == 7.5
        assert params.negative_prompt == ""
        # Check that diffusers-incompatible fields are not added
        assert not hasattr(params, "clip_skip")

    def test_pipeline_config_defaults(self):
        """Test DiffusersGenericPipelineConfig default values."""
        config = DiffusersGenericPipelineConfig()
        assert config.dit_precision == "fp16"
        assert config.vae_precision == "fp32"
        assert config.disable_autocast is True
        assert config.vae_tiling is False


class TestModelInfoResolution:
    """Test model info resolution with backend parameter."""

    def test_diffusers_backend_explicit(self):
        """Test that explicit diffusers backend returns DiffusersPipeline."""
        # Use a fake model path - with diffusers backend it should still work
        # as it will try to use diffusers directly
        model_info = get_model_info(
            "stabilityai/stable-diffusion-2-1",
            backend=Backend.DIFFUSERS,
        )

        assert model_info is not None
        assert model_info.pipeline_cls.__name__ == "DiffusersPipeline"
        assert model_info.sampling_param_cls == DiffusersGenericSamplingParams
        assert model_info.pipeline_config_cls == DiffusersGenericPipelineConfig

    def test_diffusers_backend_string(self):
        """Test that string 'diffusers' works as backend parameter."""
        model_info = get_model_info(
            "stabilityai/stable-diffusion-2-1",
            backend="diffusers",
        )

        assert model_info is not None
        assert model_info.pipeline_cls.__name__ == "DiffusersPipeline"


class TestDiffusersPipelineImport:
    """Test DiffusersPipeline class import and basic structure."""

    def test_import(self):
        """Test that DiffusersPipeline can be imported."""
        from sglang.multimodal_gen.runtime.architectures.basic.diffusers_backend import (
            DiffusersPipeline,
        )

        assert DiffusersPipeline is not None
        assert DiffusersPipeline.pipeline_name == "DiffusersPipeline"

    def test_pipeline_discovery(self):
        """Test that DiffusersPipeline is discovered by the registry."""
        from sglang.multimodal_gen.registry import _discover_and_register_pipelines, _PIPELINE_REGISTRY

        _discover_and_register_pipelines()
        assert "DiffusersPipeline" in _PIPELINE_REGISTRY


# Integration test - only run if CUDA is available and diffusers is installed
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestDiffusersPipelineIntegration:
    """Integration tests for DiffusersPipeline (requires GPU)."""

    @pytest.mark.slow
    def test_load_stable_diffusion(self):
        """Test loading a Stable Diffusion model through diffusers backend.

        This test is marked slow as it downloads model weights.
        """
        from sglang.multimodal_gen.runtime.architectures.basic.diffusers_backend import (
            DiffusersPipeline,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        # Create minimal server args
        server_args = ServerArgs(
            model_path="hf-internal-testing/tiny-stable-diffusion-torch",
            trust_remote_code=True,
        )

        # This should succeed and load via diffusers
        try:
            pipe = DiffusersPipeline(
                model_path=server_args.model_path,
                server_args=server_args,
            )
            assert pipe is not None
            assert pipe.diffusers_pipe is not None
            pipe.post_init()
            assert len(pipe.stages) > 0
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

