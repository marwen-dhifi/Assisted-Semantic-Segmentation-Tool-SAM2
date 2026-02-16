"""Image view with the mask overlay, SAM point prompts and the brush."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme
from .config import BRUSH_MAX, BRUSH_MIN, BRUSH_STEP, DEFAULT_BRUSH, OVERLAY_ALPHA, OVERLAY_COLOR

POSITIVE, NEGATIVE = 1, 0  # SAM point labels


class Canvas(QWidget):
    predict_requested = Signal(str, object, object)  # path, points (N x 2 float32), labels (N int32)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keyboard stays on the main window
        self.brush_size = DEFAULT_BRUSH
        self.mode = "point"                           # "point" | "brush"
        self.on_brush_change = None                   # callback(int)
        self.mouse_pos = None
        self._reset_state()

    def _reset_state(self):
        self.path = None
        self.rgb = None
        self.base_qimg = None
        self._base_buf = None  # QImage does not copy: the numpy buffer must outlive it
        self.h = self.w = 0
        self.mask = None
        self.overlay_qimg = None
        self._ov_buf = None
        self.points = []       # (x_img, y_img, label)
        self._painting = False
        self._erasing = False

    def set_image(self, path, rgb, mask=None):
        self.path = path
        self.rgb = rgb
        self.h, self.w = rgb.shape[:2]
        self._base_buf = np.ascontiguousarray(rgb)
        self.base_qimg = QImage(self._base_buf.data, self.w, self.h, 3 * self.w, QImage.Format.Format_RGB888)
        self.points = []
        self.mask = mask if mask is not None else np.zeros((self.h, self.w), np.uint8)
        self._painting = self._erasing = False
        self._rebuild_overlay()
        self.update()

    def clear(self):
        if self.mask is None:
            return
        self.points = []
        self.mask = np.zeros((self.h, self.w), np.uint8)
        self._rebuild_overlay()
        self.update()

    def has_mask(self) -> bool:
        return self.mask is not None and bool(self.mask.any())

    # aspect fit with letterbox offset: (scale, x offset, y offset) from image to widget pixels
    def _geom(self):
        W, H = self.width(), self.height()
        if self.w == 0:
            return 1.0, 0.0, 0.0
        s = min(W / self.w, H / self.h)
        return s, (W - self.w * s) / 2, (H - self.h * s) / 2

    def _to_img(self, px, py):
        s, ox, oy = self._geom()
        ix, iy = (px - ox) / s, (py - oy) / s
        return ix, iy, (0 <= ix < self.w and 0 <= iy < self.h)

    def _rebuild_overlay(self):
        if self.mask is None:
            self.overlay_qimg = None
            return
        rgba = np.zeros((self.h, self.w, 4), np.uint8)
        m = self.mask > 0
        rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = (*OVERLAY_COLOR, OVERLAY_ALPHA)
        self._ov_buf = np.ascontiguousarray(rgba)
        self.overlay_qimg = QImage(self._ov_buf.data, self.w, self.h, 4 * self.w, QImage.Format.Format_RGBA8888)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG_DARK))
        if self.base_qimg is None:
            p.end()
            return
        s, ox, oy = self._geom()
        target = QRectF(ox, oy, self.w * s, self.h * s)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(target, self.base_qimg)
        if self.overlay_qimg is not None:
            p.drawImage(target, self.overlay_qimg)

        for (x, y, lab) in self.points:
            col = QColor(46, 229, 46) if lab == POSITIVE else QColor(255, 64, 64)
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor("white"), 2))
            p.drawEllipse(QPointF(x * s + ox, y * s + oy), 5, 5)

        if self.mode == "brush" and self.mouse_pos is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("white"), 1, Qt.PenStyle.DashLine))
            r = self.brush_size * s
            p.drawEllipse(self.mouse_pos, r, r)
        p.end()

    def mousePressEvent(self, e):
        if self.base_qimg is None:
            return
        ix, iy, ok = self._to_img(e.position().x(), e.position().y())
        btn = e.button()
        if self.mode == "point":
            if btn == Qt.MouseButton.LeftButton and ok:
                neg = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self.points.append((ix, iy, NEGATIVE if neg else POSITIVE))
                self._run_predict()
            elif btn == Qt.MouseButton.RightButton and self.points:
                self.points.pop()
                if self.points:
                    self._run_predict()
                else:
                    self.mask = np.zeros((self.h, self.w), np.uint8)
                    self._rebuild_overlay()
                    self.update()
        else:
            if btn == Qt.MouseButton.LeftButton:
                self._painting = True
                self._paint_at(ix, iy, True)
            elif btn == Qt.MouseButton.RightButton:
                self._erasing = True
                self._paint_at(ix, iy, False)

    def mouseMoveEvent(self, e):
        self.mouse_pos = e.position()
        if self.mode == "brush":
            if self._painting or self._erasing:
                ix, iy, _ = self._to_img(e.position().x(), e.position().y())
                self._paint_at(ix, iy, self._painting)
            self.update()

    def mouseReleaseEvent(self, _):
        self._painting = self._erasing = False

    def wheelEvent(self, e):
        if self.mode != "brush":
            return
        step = BRUSH_STEP if e.angleDelta().y() > 0 else -BRUSH_STEP
        self.set_brush(self.brush_size + step)

    def set_brush(self, size: int):
        self.brush_size = max(BRUSH_MIN, min(BRUSH_MAX, size))
        if self.on_brush_change:
            self.on_brush_change(self.brush_size)
        self.update()

    def _paint_at(self, ix, iy, add):
        if self.mask is None:
            return
        cv2.circle(self.mask, (int(round(ix)), int(round(iy))), self.brush_size, 1 if add else 0, -1)
        self._rebuild_overlay()
        self.update()

    def _run_predict(self):
        # every point feeds one prediction, so an extra click refines the mask instead of replacing it
        if not self.points:
            return
        pts = np.array([[x, y] for (x, y, _) in self.points], dtype=np.float32)
        labs = np.array([lab for (_, _, lab) in self.points], dtype=np.int32)
        self.predict_requested.emit(self.path, pts, labs)

    @Slot(str, object)
    def apply_mask(self, path, mask):
        if path != self.path:
            return  # stale result for a previous image
        self.mask = mask.copy()
        self._rebuild_overlay()
        self.update()
