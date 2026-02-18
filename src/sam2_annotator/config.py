"""Model selection and annotation constants."""

from __future__ import annotations

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# overlay color per class, cycled when there are more classes than colors (RGB)
PALETTE = [
    (56, 255, 132), (79, 142, 247), (247, 79, 122), (245, 158, 11), (167, 139, 250),
    (34, 211, 238), (239, 68, 68), (163, 230, 53), (244, 114, 182), (251, 191, 36),
]
OVERLAY_ALPHA = 110             # 0-255
DEFAULT_BRUSH = 18              # radius in image pixels
BRUSH_MIN, BRUSH_MAX, BRUSH_STEP = 2, 200, 2
MAX_CLASSES = 255               # mask pixels are uint8 class ids, 0 is background
HOTKEY_CLASSES = 9              # keys 1..9


def class_color(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_model(device: str) -> str:
    # base-plus follows thin, low-contrast regions noticeably better; small keeps CPU usable
    return "facebook/sam2.1-hiera-base-plus" if device == "cuda" else "facebook/sam2.1-hiera-small"
