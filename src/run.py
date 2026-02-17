"""Open this file in an editor and run it. Change IMAGES to annotate another folder."""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent  # repo root: .venv and sample_data live here

IMAGES = ROOT / "sample_data"
OUT = None  # None: <IMAGES>_annotations next to the image folder

try:
    import PySide6  # noqa: F401
    import sam2  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    # the editor picked an interpreter without the dependencies, fall back to the repo venv
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        sys.exit(subprocess.call([str(venv_python), str(Path(__file__).resolve())]))
    sys.exit("missing dependencies, install them with: pip install -e .")

sys.path.insert(0, str(SRC))

from sam2_leak_annotator.__main__ import main  # noqa: E402

sys.exit(main([str(IMAGES)] + (["--out", str(OUT)] if OUT else [])))
