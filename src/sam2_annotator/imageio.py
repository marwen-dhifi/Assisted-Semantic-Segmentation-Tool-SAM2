"""Image and mask IO through numpy buffers: cv2.imread/imwrite fail on non-ASCII paths on Windows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_rgb(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_mask(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_png(path: Path, arr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", arr)
    if ok:
        buf.tofile(str(path))
