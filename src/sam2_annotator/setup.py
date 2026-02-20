"""Start dialog: image folder, classes and output folder. The last values are remembered."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from . import theme
from .dataset import Dataset, parse_classes

LAST_SESSION = Path.home() / ".sam2_annotator.json"


def default_out(images: Path) -> Path:
    images = images.resolve()
    return images.parent / f"{images.name}_annotations"


def _load_last() -> dict:
    try:
        return json.loads(LAST_SESSION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_last(values: dict) -> None:
    try:
        LAST_SESSION.write_text(json.dumps(values, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only home directory only costs the prefill next time


class SetupDialog(QDialog):
    def __init__(self, device: str, model: str, images: str = "", classes: str = "", out: str = ""):
        super().__init__()
        self.setWindowTitle("SAM2 Annotator")
        self.setStyleSheet(f"background:{theme.BG_PANEL};")
        self.setMinimumWidth(480)
        self.dataset: Dataset | None = None
        last = _load_last()
        images = images or last.get("images", "")
        out = out or last.get("out", "")
        # the output follows the image folder unless it was set to something other than the default
        self._out_edited = bool(out) and not (images and Path(out).resolve() == default_out(Path(images)))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(6)

        def caption(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{theme.TEXT_DIM};font:bold 11px 'Segoe UI';")
            lay.addSpacing(8)
            lay.addWidget(lbl)

        def hint(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(theme.DIM)
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            return lbl

        def entry(value, placeholder):
            e = QLineEdit(value)
            e.setPlaceholderText(placeholder)
            e.setStyleSheet(theme.ENTRY)
            return e

        def browse_row(edit, on_click):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("Browse")
            b.setStyleSheet(theme.BUTTON)
            b.clicked.connect(on_click)
            row.addWidget(b)
            lay.addLayout(row)

        title = QLabel("SAM2 ANNOTATOR")
        title.setStyleSheet(theme.TITLE)
        lay.addWidget(title)
        hint(f"device: {device}   model: {model.split('/')[-1]}")

        caption("IMAGE FOLDER")
        self.images = entry(images, "folder of images, subfolders included")
        self.images.textEdited.connect(self._images_changed)
        browse_row(self.images, self._browse_images)

        caption("CLASSES")
        self.classes = entry(classes or last.get("classes", ""), "e.g. crack, rust, dent")
        lay.addWidget(self.classes)
        hint("comma separated, in order: the first class is pixel value 1 in the masks, keys 1 to 9 select them")

        caption("OUTPUT FOLDER")
        self.out = entry(out, "masks/, classes.json and progress.json go here")
        self.out.textEdited.connect(lambda _t: setattr(self, "_out_edited", True))
        browse_row(self.out, self._browse_out)
        if not self.out.text() and self.images.text():
            self._images_changed(self.images.text())

        self.error = QLabel("")
        self.error.setStyleSheet(f"color:{theme.RED};font:12px 'Segoe UI';")
        self.error.setWordWrap(True)
        lay.addSpacing(8)
        lay.addWidget(self.error)
        start = QPushButton("Start")
        start.setStyleSheet(theme.BUTTON_PRIMARY)
        start.setDefault(True)  # Enter in any field starts
        start.clicked.connect(self._start)
        lay.addWidget(start)

    def _images_changed(self, text: str):
        # follow the image folder until the user sets an output folder of their own
        if not self._out_edited and text.strip():
            self.out.setText(str(default_out(Path(text.strip()))))

    def _browse_images(self):
        chosen = QFileDialog.getExistingDirectory(self, "Image folder", self.images.text() or str(Path.home()))
        if chosen:
            self.images.setText(chosen)
            self._images_changed(chosen)

    def _browse_out(self):
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text() or str(Path.home()))
        if chosen:
            self.out.setText(chosen)
            self._out_edited = True

    def _start(self):
        try:
            images = self.images.text().strip()
            if not images:
                raise ValueError("choose an image folder")
            classes = parse_classes(self.classes.text())
            out = self.out.text().strip() or str(default_out(Path(images)))
            self.dataset = Dataset(Path(images), Path(out), classes)
        except (ValueError, NotADirectoryError, FileNotFoundError) as e:
            self.error.setText(str(e))
            return
        _save_last({"images": images, "classes": ", ".join(classes), "out": out})
        self.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(e)
