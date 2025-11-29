# SPDX-License-Identifier: Apache-2.0
"""
SGLang attention backend integration for diffusers pipelines.

This module uses diffusers' native attention dispatcher API to set optimized
attention backends. This is cleaner and more compatible than custom processors.

See: https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
"""

from typing import Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Map SGLang/CLI backend names to diffusers backend names
_BACKEND_NAME_MAP = {
    # SGLang CLI names
    "fa": "flash",
    "fa3": "_flash_3",
    "fa4": "_flash_3",
    "flash_attn": "flash",
    "torch_sdpa": "native",
    "sdpa": "native",
    "sage_attn": "sage",
    "sage_attn_three": "_sage_qk_int8_pv_fp8_cuda",
    "sage_attn_3": "_sage_qk_int8_pv_fp8_cuda",
    # Direct diffusers names (pass through)
    "flash": "flash",
    "flash_hub": "flash_hub",
    "_flash_3": "_flash_3",
    "_flash_3_hub": "_flash_3_hub",
    "sage": "sage",
    "sage_hub": "sage_hub",
    "native": "native",
    "xformers": "xformers",
    "flex": "flex",
    "auto": "auto",
}

# Backends to try in order of preference for "auto"
_AUTO_BACKENDS = [
    "_flash_3",  # FlashAttention 3 (Hopper)
    "flash",     # FlashAttention 2
    "sage",      # SageAttention
    "native",    # PyTorch SDPA (always available)
]


def get_available_attention_backends() -> list[str]:
    """Get list of attention backends available in diffusers."""
    available = ["native"]  # Always available

    try:
        # Check if flash attention is available
        import flash_attn  # noqa: F401
        available.extend(["flash", "_flash_3"])
    except ImportError:
        pass

    try:
        from sageattention import sageattn  # noqa: F401
        available.append("sage")
    except ImportError:
        pass

    try:
        import xformers  # noqa: F401
        available.append("xformers")
    except ImportError:
        pass

    return available


def apply_sglang_attention(
    pipe: Any,
    backend: str = "auto",
) -> None:
    """
    Apply optimized attention backend to a diffusers pipeline.

    Uses diffusers' native set_attention_backend() API which handles
    all model types correctly (including video models like Wan, CogVideoX, etc.)

    Args:
        pipe: A diffusers pipeline
        backend: Attention backend. Options:
            - "auto": Auto-select best available
            - "fa", "flash": FlashAttention 2
            - "fa3", "_flash_3": FlashAttention 3
            - "sage", "sage_attn": SageAttention
            - "native", "sdpa", "torch_sdpa": PyTorch native SDPA
            - "xformers": xFormers memory-efficient attention
            - Or any diffusers backend name directly
    """
    # Map to diffusers backend name
    diffusers_backend = _BACKEND_NAME_MAP.get(backend, backend)

    # Handle auto selection
    if diffusers_backend == "auto":
        available = get_available_attention_backends()
        for candidate in _AUTO_BACKENDS:
            if candidate in available:
                diffusers_backend = candidate
                break
        logger.info("Auto-selected attention backend: %s", diffusers_backend)

    # Find models to configure
    models = []
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models.append(("transformer", pipe.transformer))
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models.append(("unet", pipe.unet))

    if not models:
        logger.debug("No transformer or unet found in pipeline")
        return

    for name, model in models:
        try:
            # Use diffusers' native API
            if hasattr(model, "set_attention_backend"):
                model.set_attention_backend(diffusers_backend)
                logger.info(
                    "Set attention backend '%s' on %s via diffusers API",
                    diffusers_backend,
                    name,
                )
            else:
                logger.info(
                    "%s does not support set_attention_backend (diffusers version may be old)",
                    name,
                )
        except Exception as e:
            logger.warning(
                "Could not set attention backend '%s' on %s: %s",
                diffusers_backend,
                name,
                e,
            )


def reset_attention_backend(pipe: Any) -> None:
    """Reset attention backend to default for all models in pipeline."""
    models = []
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        models.append(("transformer", pipe.transformer))
    if hasattr(pipe, "unet") and pipe.unet is not None:
        models.append(("unet", pipe.unet))

    for name, model in models:
        try:
            if hasattr(model, "reset_attention_backend"):
                model.reset_attention_backend()
                logger.debug("Reset attention backend on %s", name)
        except Exception as e:
            logger.warning("Could not reset attention backend on %s: %s", name, e)


# Legacy aliases for compatibility
class SGLangAttnProcessor:
    """Legacy class - use apply_sglang_attention() instead."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "SGLangAttnProcessor is deprecated. "
            "Use apply_sglang_attention() which uses diffusers' native API."
        )


SGLangAttnProcessor2_0 = SGLangAttnProcessor
