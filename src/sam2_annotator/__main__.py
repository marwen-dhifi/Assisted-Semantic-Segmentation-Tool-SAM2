"""Command line entry point. Without --classes the setup dialog asks for folder, classes and output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sam2-annotator",
                                description="SAM 2.1 assisted semantic segmentation annotation.")
    p.add_argument("images", nargs="?", help="folder of images, scanned recursively")
    p.add_argument("--classes", help="comma separated class names, e.g. crack,rust,dent (skips the setup dialog)")
    p.add_argument("--out", help="output folder for masks/, classes.json, progress.json "
                                 "(default: <images>_annotations)")
    p.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    p.add_argument("--model", help="Hugging Face id (default: sam2.1-hiera-base-plus on cuda, -small on cpu)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # torch, sam2 and Qt are imported after argument parsing so --help answers instantly
    from PySide6.QtWidgets import QApplication, QDialog

    from .config import default_device, default_model
    from .dataset import Dataset, parse_classes
    from .setup import SetupDialog, default_out
    from .window import MainWindow

    device = args.device or default_device()
    model = args.model or default_model(device)
    app = QApplication.instance() or QApplication(sys.argv)

    if args.images and args.classes:
        try:
            images = Path(args.images)
            dataset = Dataset(images, Path(args.out) if args.out else default_out(images),
                              parse_classes(args.classes))
        except (ValueError, NotADirectoryError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        dialog = SetupDialog(device, model, args.images or "", args.classes or "", args.out or "")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return 0
        dataset = dialog.dataset

    win = MainWindow(dataset, model, device)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
