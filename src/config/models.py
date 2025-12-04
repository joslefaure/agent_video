"""
Model configuration for the agentic video understanding pipeline.

This file defines the models used in each component of the pipeline.
All models are <10B parameters and use off-the-shelf pretrained weights.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    hf_model_id: str
    params: str
    description: str
    inference_settings: Dict[str, Any]


# ==============================================================================
# VLM ORCHESTRATOR (Agent Brain)
# ==============================================================================

ORCHESTRATOR_MODELS = {
    "qwen2.5-vl-7b": ModelConfig(
        name="Qwen2.5-VL-7B-Instruct",
        hf_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        params="8B",
        description="Latest Qwen VL model with excellent video understanding, "
                    "dynamic FPS sampling, and strong OCR capabilities. "
                    "Supports multi-image and video inputs natively.",
        inference_settings={
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",  # For speed
            # Video processing settings
            "min_pixels": 256 * 28 * 28,  # ~200k pixels
            "max_pixels": 1280 * 28 * 28,  # ~1M pixels
            # Generation settings
            "max_new_tokens": 1024,
            "do_sample": False,  # Deterministic for reproducibility
            "temperature": 0.0,
        }
    ),
    "internvl2.5-8b": ModelConfig(
        name="InternVL2.5-8B",
        hf_model_id="OpenGVLab/InternVL2_5-8B",
        params="8B",
        description="Strong alternative with excellent multimodal reasoning, "
                    "good video understanding, and strong performance on VideoMME.",
        inference_settings={
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "use_flash_attn": True,
            "trust_remote_code": True,
            # Generation settings
            "max_new_tokens": 1024,
            "do_sample": False,
        }
    ),
}

# Default orchestrator
DEFAULT_ORCHESTRATOR = "qwen2.5-vl-7b"


# ==============================================================================
# VISUAL ENCODER (Frame Embeddings)
# ==============================================================================

VISUAL_ENCODERS = {
    "dinov2-large": ModelConfig(
        name="DINOv2-Large",
        hf_model_id="facebook/dinov2-large",
        params="0.3B",
        description="Self-supervised vision transformer with robust features. "
                    "Industry standard for image embeddings.",
        inference_settings={
            "torch_dtype": "float32",
            "device_map": "auto",
            "trust_remote_code": False,
            # Input resolution: 518x518 by default for DINOv2-large
        }
    ),
}

DEFAULT_VISUAL_ENCODER = "dinov2-large"


# ==============================================================================
# OCR (Text Extraction)
# ==============================================================================

OCR_MODELS = {
    "got-ocr2.0": ModelConfig(
        name="GOT-OCR2.0",
        hf_model_id="stepfun-ai/GOT-OCR2_0",
        params="0.7B",
        description="State-of-the-art unified OCR model supporting plain text, "
                    "formatted text, fine-grained OCR, and multi-crop OCR.",
        inference_settings={
            "torch_dtype": "bfloat16",
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "trust_remote_code": True,
            # OCR types: 'ocr' (plain), 'format' (formatted)
            # Can use ocr_box, ocr_color for fine-grained control
        }
    ),
}

DEFAULT_OCR_MODEL = "got-ocr2.0"


# ==============================================================================
# SEGMENTATION (Masks)
# ==============================================================================

SEGMENTATION_MODELS = {
    "sam2-large": ModelConfig(
        name="SAM2-Hiera-Large",
        hf_model_id="facebook/sam2-hiera-large",
        params="0.2B",
        description="Segment Anything Model 2 from Meta. Supports image and video "
                    "segmentation with point, box, or mask prompts. "
                    "Can track objects across video frames.",
        inference_settings={
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": False,
            # Supports: single/multiple points, bounding boxes, mask refinement
            # For video: can propagate masks across frames
        }
    ),
}

DEFAULT_SEGMENTATION_MODEL = "sam2-large"


# ==============================================================================
# OBJECT DETECTION (Open-Vocabulary)
# ==============================================================================

DETECTION_MODELS = {
    "owlv2-large": ModelConfig(
        name="OWLv2-Large-Ensemble",
        hf_model_id="google/owlv2-large-patch14-ensemble",
        params="0.4B",
        description="Open-vocabulary object detection with text-conditioned queries. "
                    "CLIP-based architecture for zero-shot detection.",
        inference_settings={
            "torch_dtype": "float32",
            "device_map": "auto",
            "trust_remote_code": False,
            # Text queries: list of text descriptions for objects
            # Returns: boxes, scores, labels
        }
    ),
    "yolov8-nano": ModelConfig(
        name="YOLOv8-Nano",
        hf_model_id": "ultralytics/yolov8n",  # Loaded via ultralytics package
        params="~3M",
        description="Fast baseline detector for common COCO objects. "
                    "Useful for quick object presence checks.",
        inference_settings={
            "device": "cuda",
            "conf": 0.25,  # Confidence threshold
            "iou": 0.45,   # IoU threshold for NMS
        }
    ),
}

DEFAULT_DETECTION_MODEL = "owlv2-large"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_model_config(component: str, model_name: str = None) -> ModelConfig:
    """
    Get model configuration for a specific component.
    
    Args:
        component: One of 'orchestrator', 'encoder', 'ocr', 'segmentation', 'detection'
        model_name: Optional specific model name. If None, uses default.
    
    Returns:
        ModelConfig object
    """
    component_map = {
        "orchestrator": (ORCHESTRATOR_MODELS, DEFAULT_ORCHESTRATOR),
        "encoder": (VISUAL_ENCODERS, DEFAULT_VISUAL_ENCODER),
        "ocr": (OCR_MODELS, DEFAULT_OCR_MODEL),
        "segmentation": (SEGMENTATION_MODELS, DEFAULT_SEGMENTATION_MODEL),
        "detection": (DETECTION_MODELS, DEFAULT_DETECTION_MODEL),
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}")
    
    models_dict, default = component_map[component]
    model_key = model_name or default
    
    if model_key not in models_dict:
        raise ValueError(f"Unknown model '{model_key}' for component '{component}'")
    
    return models_dict[model_key]


def get_all_model_ids() -> Dict[str, str]:
    """Get a mapping of component -> default model HuggingFace ID."""
    return {
        "orchestrator": ORCHESTRATOR_MODELS[DEFAULT_ORCHESTRATOR].hf_model_id,
        "encoder": VISUAL_ENCODERS[DEFAULT_VISUAL_ENCODER].hf_model_id,
        "ocr": OCR_MODELS[DEFAULT_OCR_MODEL].hf_model_id,
        "segmentation": SEGMENTATION_MODELS[DEFAULT_SEGMENTATION_MODEL].hf_model_id,
        "detection": DETECTION_MODELS[DEFAULT_DETECTION_MODEL].hf_model_id,
    }


if __name__ == "__main__":
    # Print model summary
    print("=" * 80)
    print("AGENTIC VIDEO UNDERSTANDING PIPELINE - MODEL CONFIGURATION")
    print("=" * 80)
    print()
    
    for component, (models_dict, default) in [
        ("Orchestrator", (ORCHESTRATOR_MODELS, DEFAULT_ORCHESTRATOR)),
        ("Visual Encoder", (VISUAL_ENCODERS, DEFAULT_VISUAL_ENCODER)),
        ("OCR", (OCR_MODELS, DEFAULT_OCR_MODEL)),
        ("Segmentation", (SEGMENTATION_MODELS, DEFAULT_SEGMENTATION_MODEL)),
        ("Detection", (DETECTION_MODELS, DEFAULT_DETECTION_MODEL)),
    ]:
        print(f"\n{component} (Default: {default})")
        print("-" * 80)
        for key, config in models_dict.items():
            marker = "→" if key == default else " "
            print(f"{marker} {config.name} ({config.params})")
            print(f"  Model ID: {config.hf_model_id}")
            print(f"  {config.description}")
            print()
