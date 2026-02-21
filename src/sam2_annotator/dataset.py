"""Image list, class list, done-progress and label masks. Images are keyed by path relative to the image root."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import IMG_EXTS, MAX_CLASSES
from .imageio import imread_mask, imwrite_png


def parse_classes(text: str) -> list[str]:
    """Comma separated class names. Order matters: class i is stored as pixel value i + 1."""
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if not classes:
        raise ValueError("enter at least one class")
    lowered = [c.lower() for c in classes]
    if len(set(lowered)) != len(lowered):
        raise ValueError("class names must be unique")
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"at most {MAX_CLASSES} classes, mask pixels are uint8")
    return classes


class Dataset:
    """Output layout:

        <out>/masks/<rel path>.png   single channel uint8, 0 = background, k = classes[k - 1]
        <out>/classes.json           {"classes": [...]}, fixes the id of every class
        <out>/progress.json          {"done": [rel paths]}
    """

    def __init__(self, images_root: Path, out_root: Path, classes: list[str]):
        self.images_root = images_root.resolve()
        self.out_root = out_root.resolve()
        self.masks_root = self.out_root / "masks"
        self.classes_json = self.out_root / "classes.json"
        self.progress_json = self.out_root / "progress.json"
        self.classes = list(classes)
        if not self.images_root.is_dir():
            raise NotADirectoryError(f"image folder not found: {self.images_root}")

        # an output folder inside the image folder must not feed its own masks back as images
        self.images = sorted(p for p in self.images_root.rglob("*")
                             if p.suffix.lower() in IMG_EXTS and not p.is_relative_to(self.out_root))
        if not self.images:
            raise FileNotFoundError(f"no images found under {self.images_root}")
        self._check_classes()
        self.done = self._load_progress()

    def _check_classes(self) -> None:
        # masks store ids, not names: reusing an output folder with another list would silently relabel them
        if self.classes_json.exists():
            try:
                saved = json.loads(self.classes_json.read_text(encoding="utf-8")).get("classes", [])
            except (OSError, ValueError):
                saved = None
            if saved and saved != self.classes:
                raise ValueError(f"{self.out_root} was annotated with classes {', '.join(saved)}; "
                                 "use the same list in the same order, or another output folder")
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.classes_json.write_text(json.dumps({"classes": self.classes}, indent=2), encoding="utf-8")

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

    def load_label(self, img_path) -> np.ndarray | None:
        mp = self.mask_path(img_path)
        if not mp.exists():
            return None
        m = imread_mask(str(mp))
        if m is None:
            return None
        return m if m.ndim == 2 else m[..., 0]

    def commit(self, img_path, label: np.ndarray) -> bool:
        """Save the label map of an image being left and set its done state. Returns True if annotated.

        An image is annotated exactly when its mask has foreground: a non-empty mask is written and
        marked done, an empty one removes any old mask file and the done mark, so a mask cleared by
        the user does not survive on disk.
        """
        mp = self.mask_path(img_path)
        rel = self.rel(img_path)
        annotated = bool(label.any())
        if annotated:
            mp.parent.mkdir(parents=True, exist_ok=True)
            imwrite_png(mp, label.astype(np.uint8))
        elif mp.exists():
            mp.unlink()
        if annotated != (rel in self.done):
            (self.done.add if annotated else self.done.discard)(rel)
            self.save_progress()
        return annotated

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
