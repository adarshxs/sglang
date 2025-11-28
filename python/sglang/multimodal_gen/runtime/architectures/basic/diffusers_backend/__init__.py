# SPDX-License-Identifier: Apache-2.0
"""Diffusers backend for running vanilla diffusers pipelines through sglang."""

from sglang.multimodal_gen.runtime.architectures.basic.diffusers_backend.diffusers_pipeline import (
    DiffusersPipeline,
)

__all__ = ["DiffusersPipeline"]

