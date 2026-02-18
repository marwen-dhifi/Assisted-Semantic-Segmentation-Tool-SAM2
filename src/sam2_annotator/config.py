"""Model selection and annotation constants."""

from __future__ import annotations

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

OVERLAY_COLOR = (56, 255, 132)  # RGB
OVERLAY_ALPHA = 110             # 0-255
DEFAULT_BRUSH = 18              # radius in image pixels
BRUSH_MIN, BRUSH_MAX, BRUSH_STEP = 2, 200, 2


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_model(device: str) -> str:
    # base-plus is noticeably better on thin, low-contrast water films; small keeps CPU usable
    return "facebook/sam2.1-hiera-base-plus" if device == "cuda" else "facebook/sam2.1-hiera-small"
