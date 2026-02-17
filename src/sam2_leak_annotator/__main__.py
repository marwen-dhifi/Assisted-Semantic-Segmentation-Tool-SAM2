"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sam2-leak-annotator",
                                description="SAM 2.1 assisted binary mask annotation for leak images.")
    p.add_argument("images", help="folder of images, scanned recursively")
    p.add_argument("--out", help="output folder for masks/ and progress.json (default: <images>_annotations)")
    p.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    p.add_argument("--model", help="Hugging Face id (default: sam2.1-hiera-base-plus on cuda, -small on cpu)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = Path(args.images)
    out = Path(args.out) if args.out else images.resolve().parent / f"{images.resolve().name}_annotations"

    # torch, sam2 and Qt are imported after argument parsing so --help answers instantly
    from PySide6.QtWidgets import QApplication

    from .config import default_device, default_model
    from .dataset import Dataset
    from .window import MainWindow

    try:
        dataset = Dataset(images, out)
    except (NotADirectoryError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    device = args.device or default_device()
    model = args.model or default_model(device)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(dataset, model, device)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
