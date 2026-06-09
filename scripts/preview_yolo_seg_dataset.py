#!/usr/bin/env python3
"""Preview YOLO segmentation labels as polygons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(data_yaml: Path) -> list[str]:
    names_line = ""
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names_line = line.split(":", 1)[1].strip()
            break
    if not names_line:
        return []
    stripped = names_line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return [part.strip().strip("'\"") for part in stripped[1:-1].split(",") if part.strip()]
    return []


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    root = dataset_root / "images" / split
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def make_contact_sheet(images: list[np.ndarray], columns: int, thumb_width: int) -> np.ndarray:
    thumbs = []
    for image in images:
        scale = thumb_width / image.shape[1]
        size = (thumb_width, max(1, int(round(image.shape[0] * scale))))
        thumbs.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))

    rows = []
    for start in range(0, len(thumbs), columns):
        row_images = thumbs[start : start + columns]
        max_height = max(image.shape[0] for image in row_images)
        padded = []
        for image in row_images:
            if image.shape[0] < max_height:
                pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
                image = np.vstack([image, pad])
            padded.append(image)
        while len(padded) < columns:
            padded.append(np.zeros((max_height, thumb_width, 3), dtype=np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def read_seg_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts[1:]) % 2 != 0:
            continue
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        points = np.asarray(
            [[coords[index] * width, coords[index + 1] * height] for index in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        labels.append((class_id, points))
    return labels


def draw_labels(image: np.ndarray, labels: list[tuple[int, np.ndarray]], names: list[str]) -> None:
    overlay = image.copy()
    for index, (class_id, points) in enumerate(labels, start=1):
        cv2.fillPoly(overlay, [points], color=(0, 255, 255))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        center = points.mean(axis=0).astype(int)
        label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        cv2.circle(image, tuple(center), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"{index}: {label}",
            (int(points[:, 0].min()), max(18, int(points[:, 1].min()) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if labels:
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("yolo_seg_dataset/data.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=360)
    args = parser.parse_args()

    dataset_root = args.data.parent
    names = read_names(args.data)
    label_dir = dataset_root / "labels" / args.split
    paths = image_paths(dataset_root, args.split)

    rendered = []
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_seg_labels(label_path, image.shape[1], image.shape[0])
        draw_labels(image, labels, names)
        rendered.append(image)
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_dir / image_path.name), image)

    if args.contact_sheet and rendered:
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet = make_contact_sheet(rendered, args.columns, args.thumb_width)
        cv2.imwrite(str(args.contact_sheet), sheet)
        print(f"Wrote {args.contact_sheet}")

    print(f"Previewed {len(rendered)} {args.split} images from {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
