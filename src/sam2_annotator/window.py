"""Main window: left control panel, canvas, and the wiring to the SAM worker thread."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from . import theme
from .canvas import Canvas
from .dataset import Dataset
from .imageio import imread_rgb
from .worker import Sam2Worker

HELP = ("Left = leak point\nShift+Left = NOT-leak point\nRight = undo last point\n"
        "Brush mode: Left paint / Right erase\nWheel or [ ] = brush size\n"
        "Delete = remove image+mask from dataset")


class MainWindow(QMainWindow):
    request_encode = Signal(str, object)

    def __init__(self, dataset: Dataset, model_id: str, device: str, worker_cls=Sam2Worker):
        super().__init__()
        self.setWindowTitle("Leak Annotator | SAM2")
        self.resize(1400, 880)
        self.setStyleSheet(f"background:{theme.BG_DARK};")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.ds = dataset
        self.model_id = model_id
        self.device = device
        self.idx = dataset.first_unlabeled()
        self._model_ready = False

        self._build_ui()
        self._start_worker(worker_cls)

    # UI

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        left = QFrame()
        left.setFixedWidth(320)
        left.setStyleSheet(f"background:{theme.BG_PANEL};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(20, 20, 20, 20)
        lv.setSpacing(12)

        def label(text, style, wrap=False):
            lbl = QLabel(text)
            lbl.setStyleSheet(style)
            lbl.setWordWrap(wrap)
            return lbl

        def button(text, cb, style=theme.BUTTON):
            b = QPushButton(text)
            b.setStyleSheet(style)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space must reach the window, not a focused button
            b.clicked.connect(cb)
            return b

        lv.addWidget(label("LEAK ANNOTATOR", theme.TITLE))
        lv.addWidget(label(f"device: {self.device}   model: {self.model_id.split('-')[-1]}", theme.DIM, True))
        self.stats_lbl = label("", theme.DIM, True)
        lv.addWidget(self.stats_lbl)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(theme.PROGRESS)
        lv.addWidget(self.progress)

        jrow = QHBoxLayout()
        jrow.addWidget(label("Jump to #", theme.DIM))
        self.jump = QLineEdit()
        self.jump.setPlaceholderText("e.g. 458")
        self.jump.setStyleSheet(theme.ENTRY)
        self.jump.returnPressed.connect(self._do_jump)
        jrow.addWidget(self.jump)
        lv.addLayout(jrow)

        self.mode_lbl = label("", theme.MODE)
        lv.addWidget(self.mode_lbl)
        lv.addWidget(button("Toggle Brush / Point  (B)", self._toggle_mode))
        lv.addWidget(button("Clear mask  (C)", lambda: self.canvas.clear()))
        lv.addWidget(button("Next unlabeled", self._next_unlabeled))
        nav = QHBoxLayout()
        nav.addWidget(button("← Prev (Q)", self._prev))
        nav.addWidget(button("Next (D) →", self._next))
        lv.addLayout(nav)
        lv.addWidget(button("DONE + Save  (Space)", self._done_and_next, theme.BUTTON_PRIMARY))
        lv.addWidget(button("Delete image  (Delete)", self._delete_current, theme.BUTTON_DANGER))
        lv.addWidget(label(HELP, theme.DIM, True))
        lv.addStretch(1)
        self.status_lbl = label("", theme.DIM, True)
        lv.addWidget(self.status_lbl)

        self.canvas = Canvas()
        self.canvas.on_brush_change = lambda _size: self._refresh_mode_label()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay.addWidget(left)
        lay.addWidget(self.canvas, 1)
        self._refresh_mode_label()
        self._refresh_stats()

    def _status(self, text: str, warn: bool = False):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(theme.WARN if warn else theme.DIM)

    # worker

    def _start_worker(self, worker_cls):
        self.thread = QThread()
        self.worker = worker_cls(self.model_id, self.device)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.init_model)
        self.worker.encoded.connect(self._on_encoded)
        self.worker.mask_ready.connect(self.canvas.apply_mask)
        self.worker.error.connect(self._on_error)
        self.canvas.predict_requested.connect(self.worker.predict)
        self.request_encode.connect(self.worker.encode)
        self._status("loading SAM2 model...")
        self.thread.start()

    @Slot(str)
    def _on_encoded(self, token):
        if token == "__ready__":
            self._model_ready = True
            self._show(self.idx)
        elif token == self._cur_path():
            self._status("ready, click on the leak")

    @Slot(str)
    def _on_error(self, msg):
        self._status(msg, warn=True)

    # navigation

    def _cur_path(self) -> str:
        return str(self.ds.images[self.idx])

    def _show(self, idx, autosave=True):
        if autosave and self.canvas.path is not None:
            self._save_mask(self.canvas.path)
        self.idx = max(0, min(idx, len(self.ds.images) - 1))
        path = self._cur_path()
        rgb = imread_rgb(path)
        if rgb is None:
            self._status(f"cannot read {Path(path).name}", warn=True)
            return
        self.canvas.set_image(path, rgb, self.ds.load_mask(path))
        if self._model_ready:
            self._status("encoding image...")
            self.request_encode.emit(path, rgb)
        self._refresh_stats()

    def _save_mask(self, path):
        if self.canvas.path != path or self.canvas.mask is None:
            return
        self.ds.save_mask(path, self.canvas.mask)

    def _prev(self):
        if self.idx > 0:
            self._show(self.idx - 1)

    def _next(self):
        if self.idx < len(self.ds.images) - 1:
            self._show(self.idx + 1)

    def _done_and_next(self):
        path = self.canvas.path
        if path is not None:
            if not self.canvas.has_mask():
                self._status("empty mask, not marked done", warn=True)
                return
            self._save_mask(path)
            self.ds.mark_done(path)
        self._next()

    def _next_unlabeled(self):
        i = self.ds.next_unlabeled(self.idx)
        if i is None:
            self._status("no unlabeled images after this one")
        else:
            self._show(i)

    def _delete_current(self):
        if not self.ds.images or self.canvas.path is None:
            return
        rel = self.ds.rel(self._cur_path())
        resp = QMessageBox.question(
            self, "Delete image",
            f"Permanently delete this image AND its mask?\n\n{rel}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            self.ds.delete(self.idx)
        except OSError as e:
            self._on_error(f"could not delete image: {e}")
            return

        if not self.ds.images:
            self.canvas._reset_state()
            self.canvas.update()
            self._status("no images left")
            self._refresh_stats()
            return
        # idx now points at the image that followed the deleted one; no autosave, the canvas
        # still references the deleted path
        self.idx = min(self.idx, len(self.ds.images) - 1)
        self._show(self.idx, autosave=False)

    def _do_jump(self):
        txt = self.jump.text().strip()
        self.jump.clear()
        self.setFocus()
        if txt.isdigit():
            self._show(int(txt) - 1)  # 1-based

    # labels

    def _toggle_mode(self):
        self.canvas.mode = "brush" if self.canvas.mode == "point" else "point"
        self._refresh_mode_label()
        self.canvas.update()

    def _refresh_mode_label(self):
        extra = f"   brush={self.canvas.brush_size}" if self.canvas.mode == "brush" else ""
        self.mode_lbl.setText(f"MODE: {self.canvas.mode.upper()}{extra}")

    def _refresh_info(self):
        if not self.ds.images:
            self.setWindowTitle("Leak Annotator | SAM2")
            return
        rel = self.ds.rel(self._cur_path())
        tag = "done" if rel in self.ds.done else "open"
        self.setWindowTitle(f"[{self.idx + 1}/{len(self.ds.images)}] {rel}  {tag}")

    def _refresh_stats(self):
        d, t = len(self.ds.done), len(self.ds.images)
        self.stats_lbl.setText(f"{d} done   ·   {t - d} left   ·   {t} total")
        self.progress.setMaximum(max(t, 1))
        self.progress.setValue(d)
        self._refresh_info()

    # keys

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Space:
            self._done_and_next()
        elif k == Qt.Key.Key_D:
            self._next()
        elif k == Qt.Key.Key_Q:
            self._prev()
        elif k == Qt.Key.Key_B:
            self._toggle_mode()
        elif k == Qt.Key.Key_C:
            self.canvas.clear()
        elif k == Qt.Key.Key_Delete:
            self._delete_current()
        elif k == Qt.Key.Key_BracketLeft:
            self.canvas.set_brush(self.canvas.brush_size - 2)
        elif k == Qt.Key.Key_BracketRight:
            self.canvas.set_brush(self.canvas.brush_size + 2)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.canvas.path is not None:
            self._save_mask(self.canvas.path)
        self.ds.save_progress()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(e)
