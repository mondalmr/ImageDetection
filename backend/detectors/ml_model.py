from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "efficientnet_b0_forgery.pth"
MODEL_NAME = "efficientnet_b0"

_model: torch.nn.Module | None = None
_device: torch.device | None = None

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> torch.nn.Module | None:
    global _model, _device

    if _model is not None:
        return _model

    if not WEIGHTS_PATH.exists():
        return None

    _device = _get_device()
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)

    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=_device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception:
        return None

    model.eval()
    model.to(_device)
    _model = model
    return _model


def analyze_ml_model(image: Image.Image) -> tuple[float | None, list[str], dict[str, Any]]:
    flags: list[str] = []
    model = _load_model()

    if model is None:
        flags.append("model_weights_unavailable")
        return None, flags, {"weights_path": str(WEIGHTS_PATH), "model": MODEL_NAME}

    device = _device or _get_device()
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    morph_probability = float(probabilities[1])
    original_probability = float(probabilities[0])

    if morph_probability > 0.7:
        flags.append("high_ml_morph_confidence")
    elif morph_probability < 0.3:
        flags.append("high_ml_original_confidence")

    details = {
        "model": MODEL_NAME,
        "morph_probability": round(morph_probability, 4),
        "original_probability": round(original_probability, 4),
        "device": str(device),
    }

    return morph_probability, flags, details
