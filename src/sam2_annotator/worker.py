"""SAM 2.1 predictor on its own QThread, so image encoding never blocks the UI."""

from __future__ import annotations

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot


class Sam2Worker(QObject):
    encoded = Signal(str)             # path that finished encoding, or "__ready__" once the model is loaded
    mask_ready = Signal(str, int, object)  # path, class index, mask (uint8 HxW, values {0, 1})
    error = Signal(str)

    def __init__(self, model_id: str, device: str):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.predictor = None
        self.cur_path = None

    @Slot()
    def init_model(self):
        try:
            # imported here: loading sam2 pulls in hydra and the model registry, which is slow
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=self.device)
            self.encoded.emit("__ready__")
        except Exception as e:
            self.error.emit(f"Model load failed: {e}")

    @Slot(str, object)
    def encode(self, path: str, rgb: object):
        # the image embedding is computed once per image; every click afterwards only runs the decoder
        if self.predictor is None:
            return
        try:
            with torch.inference_mode():
                self.predictor.set_image(rgb)
            self.cur_path = path
            self.encoded.emit(path)
        except Exception as e:
            self.error.emit(f"Encode failed: {e}")

    @Slot(str, int, object, object)
    def predict(self, path: str, cls: int, pts: object, labels: object):
        # points for an image that is no longer the encoded one would be decoded against the wrong embedding
        if self.predictor is None or path != self.cur_path:
            return
        try:
            with torch.inference_mode():
                masks, _scores, _ = self.predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=False,
                )
            self.mask_ready.emit(path, cls, (masks[0] > 0).astype(np.uint8))
        except Exception as e:
            self.error.emit(f"Predict failed: {e}")
