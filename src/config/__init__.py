"""Configuration module for the video understanding pipeline."""

from .models import (
    ModelConfig,
    get_model_config,
    get_all_model_ids,
    DEFAULT_ORCHESTRATOR,
    DEFAULT_VISUAL_ENCODER,
    DEFAULT_OCR_MODEL,
    DEFAULT_SEGMENTATION_MODEL,
    DEFAULT_DETECTION_MODEL,
)

__all__ = [
    "ModelConfig",
    "get_model_config",
    "get_all_model_ids",
    "DEFAULT_ORCHESTRATOR",
    "DEFAULT_VISUAL_ENCODER",
    "DEFAULT_OCR_MODEL",
    "DEFAULT_SEGMENTATION_MODEL",
    "DEFAULT_DETECTION_MODEL",
]
