#!/usr/bin/env python3
"""Train a YOLO segmentation model on the exported OAK-D SR dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("yolo_seg_dataset/data.yaml"))
    parser.add_argument("--model", default="/home/johannes/projects/oak_camera/yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("training_runs"))
    parser.add_argument("--name", default="knopp_oakdsr_yolo_seg")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

    from ultralytics import YOLO

    if not args.data.exists():
        parser.error(f"missing dataset config: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        cache=False,
        plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
