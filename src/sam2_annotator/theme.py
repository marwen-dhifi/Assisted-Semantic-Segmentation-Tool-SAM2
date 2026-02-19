"""Colors and Qt style sheets."""

BG_DARK = "#0f1117"
BG_PANEL = "#1a1d27"
BG_CARD = "#22263a"
BG_HOVER = "#2d3250"
ACCENT = "#4f8ef7"
ACCENT_HOVER = "#3a6fd8"
TEXT_MAIN = "#e8eaf0"
TEXT_DIM = "#6b7280"
GREEN = "#22c55e"
YELLOW = "#f59e0b"
RED = "#ef4444"

TITLE = f"color:{ACCENT};font:bold 20px 'Consolas';"
DIM = f"color:{TEXT_DIM};font:12px 'Segoe UI';"
WARN = f"color:{YELLOW};font:12px 'Segoe UI';"
MODE = f"color:{TEXT_MAIN};font:bold 13px 'Segoe UI';"
ENTRY = (f"background:{BG_CARD};color:{TEXT_MAIN};border:none;"
         f"border-radius:6px;padding:6px;font:13px 'Consolas';")
PROGRESS = (f"QProgressBar{{background:{BG_CARD};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}")
BUTTON = (f"QPushButton{{background:{BG_CARD};color:{TEXT_MAIN};border:none;"
          f"border-radius:8px;padding:10px;font:13px 'Segoe UI';}}"
          f"QPushButton:hover{{background:{BG_HOVER};}}")
BUTTON_PRIMARY = (f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:8px;"
                  f"padding:12px;font:bold 14px 'Segoe UI';}}QPushButton:hover{{background:{ACCENT_HOVER};}}")
BUTTON_DANGER = (f"QPushButton{{background:#2d1a1a;color:{RED};border:none;border-radius:8px;"
                 f"padding:10px;font:13px 'Segoe UI';}}QPushButton:hover{{background:#4a2020;}}")


def class_button(color: tuple[int, int, int], active: bool) -> str:
    hexc = "#%02x%02x%02x" % color
    if active:
        return (f"QPushButton{{background:{BG_HOVER};color:{TEXT_MAIN};border:2px solid {hexc};"
                f"border-radius:8px;padding:8px;text-align:left;font:bold 13px 'Consolas';}}")
    return (f"QPushButton{{background:{BG_CARD};color:{TEXT_MAIN};border:none;border-left:4px solid {hexc};"
            f"border-radius:8px;padding:8px;text-align:left;font:13px 'Consolas';}}"
            f"QPushButton:hover{{background:{BG_HOVER};}}")
