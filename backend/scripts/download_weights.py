"""Download or initialize EfficientNet-B0 forgery detection weights."""

from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "efficientnet_b0_forgery.pth"
MODEL_NAME = "efficientnet_b0"

# Public timm ImageNet pretrained backbone used as a starting checkpoint.
# Replace with fine-tuned forgery weights when available for production use.
FORGERY_WEIGHTS_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/"
    "efficientnet_b0_ra-3dd342df.pth"
)


def _download_backbone_weights() -> dict:
    """Load ImageNet backbone weights via timm hub (cached automatically)."""
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000)
    return model.state_dict()


def _build_forgery_model_state() -> dict:
    """
    Build a binary classifier checkpoint.

    Uses ImageNet-pretrained backbone weights and initializes a fresh
    2-class head. For production, replace this file with weights fine-tuned
    on CASIA/CoMoFoD or similar forgery datasets.
    """
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
    return model.state_dict()


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if WEIGHTS_PATH.exists():
        print(f"Weights already exist at {WEIGHTS_PATH}")
        return 0

    print(f"Creating forgery detection weights at {WEIGHTS_PATH} ...")
    print("Note: Using ImageNet-pretrained EfficientNet-B0 with a fresh binary head.")
    print("For best accuracy, replace with weights fine-tuned on forgery datasets.")

    state_dict = _build_forgery_model_state()
    torch.save(state_dict, WEIGHTS_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
