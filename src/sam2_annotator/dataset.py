"""Image list, done-progress and mask files. Everything is keyed by path relative to the image root."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import IMG_EXTS
from .imageio import imread_mask, imwrite_png


class Dataset:
    """Output layout:

        <out>/masks/<rel path>.png   single channel, values {0, 1}
        <out>/progress.json          {"done": [rel paths]}
    """

    def __init__(self, images_root: Path, out_root: Path):
        self.images_root = images_root.resolve()
        self.out_root = out_root.resolve()
        self.masks_root = self.out_root / "masks"
        self.progress_json = self.out_root / "progress.json"
        if not self.images_root.is_dir():
            raise NotADirectoryError(f"image folder not found: {self.images_root}")

        # an output folder inside the image folder must not feed its own masks back as images
        self.images = sorted(p for p in self.images_root.rglob("*")
                             if p.suffix.lower() in IMG_EXTS and not p.is_relative_to(self.out_root))
        if not self.images:
            raise FileNotFoundError(f"no images found under {self.images_root}")
        self.done = self._load_progress()

    def rel(self, path) -> str:
        return Path(path).relative_to(self.images_root).as_posix()

    def mask_path(self, img_path) -> Path:
        return self.masks_root / Path(img_path).relative_to(self.images_root).with_suffix(".png")

    def _load_progress(self) -> set[str]:
        if self.progress_json.exists():
            try:
                return set(json.loads(self.progress_json.read_text(encoding="utf-8")).get("done", []))
            except (OSError, ValueError):
                return set()
        return set()

    def save_progress(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.progress_json.write_text(json.dumps({"done": sorted(self.done)}, indent=2), encoding="utf-8")

    def first_unlabeled(self) -> int:
        for i, p in enumerate(self.images):
            if self.rel(p) not in self.done:
                return i
        return 0

    def next_unlabeled(self, after: int) -> int | None:
        for i in range(after + 1, len(self.images)):
            if self.rel(self.images[i]) not in self.done:
                return i
        return None

    def load_mask(self, img_path) -> np.ndarray | None:
        mp = self.mask_path(img_path)
        if not mp.exists():
            return None
        m = imread_mask(str(mp))
        return None if m is None else (m > 0).astype(np.uint8)

    def save_mask(self, img_path, mask: np.ndarray) -> None:
        # an empty mask is not written: "no file" and "not annotated yet" stay the same thing
        if mask.any():
            mp = self.mask_path(img_path)
            mp.parent.mkdir(parents=True, exist_ok=True)
            imwrite_png(mp, mask)  # uint8, values {0, 1}

    def mark_done(self, img_path) -> None:
        self.done.add(self.rel(img_path))
        self.save_progress()

    def delete(self, idx: int) -> None:
        """Remove the image file, its mask and its progress entry. Raises OSError if the image cannot go."""
        path = self.images[idx]
        rel = self.rel(path)
        if path.exists():
            path.unlink()
        mp = self.mask_path(path)
        try:
            if mp.exists():
                mp.unlink()
        except OSError:
            pass
        self.done.discard(rel)
        del self.images[idx]
        self.save_progress()
