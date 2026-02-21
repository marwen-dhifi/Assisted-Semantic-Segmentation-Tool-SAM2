"""Collapsible shortcut card pinned to the top right corner of the canvas."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme

ROWS = [
    ("1 - 9", "active class"),
    ("Left click", "object point"),
    ("Shift + Left", "exclude point"),
    ("Right click", "undo last point"),
    ("B", "point / brush mode"),
    ("Left / Right drag", "paint / erase in brush mode"),
    ("Wheel,  [  ]", "brush size"),
    ("C", "clear the active class"),
    ("Space", "done, save and next"),
    ("D  /  Q", "next / previous image"),
    ("Delete", "remove image and mask"),
    ("H", "show / hide this card"),
]
MARGIN = 16  # distance from the canvas edges, px

CARD = (f"#card{{background:rgba(15,17,23,225);border:1px solid {theme.BG_HOVER};border-radius:12px;}}")
HEADER = (f"QPushButton{{background:transparent;color:{theme.TEXT_MAIN};border:none;text-align:left;"
          f"padding:8px 12px;font:bold 12px 'Segoe UI';}}QPushButton:hover{{color:{theme.ACCENT};}}")
KEYCAP = (f"background:{theme.BG_CARD};color:{theme.TEXT_MAIN};border:1px solid #3a4060;"
          f"border-bottom:2px solid #3a4060;border-radius:5px;padding:2px 7px;font:11px 'Consolas';")
ACTION = f"color:{theme.TEXT_DIM};font:12px 'Segoe UI';padding-left:4px;"


class ShortcutsCard(QFrame):
    def __init__(self, canvas: QWidget):
        super().__init__(canvas)
        self.setObjectName("card")
        self.setStyleSheet(CARD)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 6)
        lay.setSpacing(0)
        self.header = QPushButton()
        self.header.setStyleSheet(HEADER)
        self.header.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keys must keep going to the main window
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.clicked.connect(self.toggle)
        lay.addWidget(self.header)

        self.body = QWidget()
        self.body.setStyleSheet("background:transparent;")
        grid = QGridLayout(self.body)
        grid.setContentsMargins(12, 2, 12, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        for r, (keys, action) in enumerate(ROWS):
            key = QLabel(keys)
            key.setStyleSheet(KEYCAP)
            grid.addWidget(key, r, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            text = QLabel(action)
            text.setStyleSheet(ACTION)
            grid.addWidget(text, r, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.body)

        canvas.installEventFilter(self)  # follow the canvas when the window is resized
        self.set_open(False)

    def set_open(self, open_: bool):
        self.body.setVisible(open_)
        self.header.setText("Shortcuts          hide  H" if open_ else "Shortcuts   H")
        self.adjustSize()
        self._place()
        self.raise_()

    def toggle(self):
        self.set_open(not self.body.isVisible())

    @property
    def is_open(self) -> bool:
        return self.body.isVisible()

    def _place(self):
        parent = self.parentWidget()
        self.move(parent.width() - self.width() - MARGIN, MARGIN)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize:
            self._place()
        return False
