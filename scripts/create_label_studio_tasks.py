#!/usr/bin/env python3
"""Create Label Studio import tasks for captured segmentation images."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
from pathlib import Path


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, default=Path("segmentation_dataset"))
    parser.add_argument("--output", type=Path, default=Path("segmentation_dataset/label_studio_tasks_embedded.json"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--embed", action="store_true", help="Embed image bytes as data URIs")
    args = parser.parse_args()

    metadata_path = args.prepared_dir / "metadata.csv"
    if not metadata_path.exists():
        parser.error(f"missing metadata file: {metadata_path}")

    tasks = []
    for row in read_metadata(metadata_path):
        if args.split != "all" and row["split"] != args.split:
            continue

        image_path = args.prepared_dir / row["filename"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if args.embed:
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            image_value = f"data:{mime_type};base64,{encoded}"
        else:
            image_value = f"/data/local-files/?d={row['filename']}"

        tasks.append(
            {
                "data": {
                    "image": image_value,
                    "filename": row["filename"],
                    "object_class": row.get("object_class", ""),
                    "split": row.get("split", ""),
                    "camera_socket": row.get("camera_socket", ""),
                    "note": row.get("note", ""),
                    "original_filename": Path(row["filename"]).name,
                }
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} Label Studio tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
