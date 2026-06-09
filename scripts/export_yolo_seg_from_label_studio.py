#!/usr/bin/env python3
"""Export Label Studio polygon annotations to a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetadataRow:
    filename: str
    object_class: str
    split: str


@dataclass(frozen=True)
class TaskAnnotations:
    task_id: int
    data: dict[str, object]
    polygons: list[dict[str, object]]


def find_project(cursor: sqlite3.Cursor, project: str | None) -> tuple[int, str]:
    if project is None:
        rows = cursor.execute("select id, title from project order by id").fetchall()
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one project, found {len(rows)}. Use --project.")
        return int(rows[0][0]), str(rows[0][1])

    if project.isdigit():
        row = cursor.execute("select id, title from project where id = ?", (int(project),)).fetchone()
    else:
        row = cursor.execute("select id, title from project where title = ?", (project,)).fetchone()
    if row is None:
        raise ValueError(f"Could not find Label Studio project: {project}")
    return int(row[0]), str(row[1])


def write_data_yaml(output_dir: Path, labels: list[str]) -> None:
    names = ", ".join(f"'{label}'" for label in labels)
    content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: [{names}]\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def read_metadata_rows(path: Path) -> list[MetadataRow]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        MetadataRow(
            filename=row["filename"],
            object_class=row.get("object_class", ""),
            split=row.get("split", "train") or "train",
        )
        for row in rows
    ]


def collect_polygon_annotations(cursor: sqlite3.Cursor, project_id: int) -> list[TaskAnnotations]:
    rows = cursor.execute(
        """
        select t.id,
               t.data,
               (
                 select a.result
                 from task_completion a
                 where a.task_id = t.id
                   and a.was_cancelled = 0
                 order by a.id desc
                 limit 1
               ) as result
        from task t
        where t.project_id = ?
        order by t.id
        """,
        (project_id,),
    ).fetchall()

    annotations = []
    for task_id, data_json, result_json in rows:
        data = json.loads(data_json)
        result = json.loads(result_json or "[]")
        polygons = [item for item in result if item.get("type") == "polygonlabels"]
        annotations.append(TaskAnnotations(int(task_id), data, polygons))
    return annotations


def label_names_from_polygons(annotations: list[TaskAnnotations]) -> list[str]:
    labels = set()
    for task in annotations:
        for polygon in task.polygons:
            value = polygon.get("value", {})
            for label in value.get("polygonlabels", []):
                labels.add(str(label))
    return sorted(labels)


def metadata_for_task(task: TaskAnnotations, index: int, metadata_rows: list[MetadataRow]) -> MetadataRow:
    data_filename = str(task.data.get("filename") or task.data.get("original_filename") or "")
    if data_filename:
        for row in metadata_rows:
            if row.filename == data_filename or Path(row.filename).name == Path(data_filename).name:
                return row
        raise KeyError(f"No metadata row matches task {task.task_id}: {data_filename}")

    if index >= len(metadata_rows):
        raise IndexError(
            f"Task {task.task_id} has no filename, and metadata only has {len(metadata_rows)} rows."
        )
    return metadata_rows[index]


def output_splits(records: list[tuple[TaskAnnotations, MetadataRow]], val_ratio: float, seed: int) -> dict[int, str]:
    metadata_splits = {metadata.split for _task, metadata in records}
    if len(metadata_splits) > 1 or "val" in metadata_splits:
        return {task.task_id: metadata.split for task, metadata in records}

    count = len(records)
    if count <= 1 or val_ratio <= 0:
        return {task.task_id: "train" for task, _metadata in records}

    val_count = max(1, round(count * val_ratio))
    val_count = min(val_count, count - 1)
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_count])
    return {
        task.task_id: ("val" if index in val_indices else "train")
        for index, (task, _metadata) in enumerate(records)
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_to_yolo_line(polygon: dict[str, object], label_to_id: dict[str, int]) -> str | None:
    value = polygon.get("value", {})
    labels = value.get("polygonlabels", [])
    points = value.get("points", [])
    if not labels or len(points) < 3:
        return None

    label = str(labels[0])
    class_id = label_to_id[label]
    coords = []
    for point in points:
        x_percent, y_percent = point
        coords.append(clamp01(float(x_percent) / 100.0))
        coords.append(clamp01(float(y_percent) / 100.0))

    return f"{class_id} " + " ".join(f"{coord:.6f}" for coord in coords)


def export_yolo_seg(args: argparse.Namespace) -> None:
    metadata_rows = read_metadata_rows(args.prepared_dir / "metadata.csv")
    connection = sqlite3.connect(args.label_studio_db)
    cursor = connection.cursor()
    project_id, project_title = find_project(cursor, args.project)
    annotations = collect_polygon_annotations(cursor, project_id)

    if len(annotations) != len(metadata_rows):
        print(
            f"Warning: Label Studio has {len(annotations)} tasks, "
            f"metadata has {len(metadata_rows)} rows. Matching by task order where needed."
        )

    labels = label_names_from_polygons(annotations)
    if not labels:
        raise ValueError("No polygon labels found in Label Studio annotations.")
    label_to_id = {label: index for index, label in enumerate(labels)}

    task_metadata = [
        (task, metadata_for_task(task, index, metadata_rows))
        for index, task in enumerate(annotations)
    ]
    split_by_task = output_splits(task_metadata, args.val_ratio, args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise ValueError(f"{args.output_dir} already exists; use --overwrite")
        shutil.rmtree(args.output_dir)

    counts: Counter[tuple[str, str]] = Counter()
    copied = 0
    for task, metadata in task_metadata:
        split = split_by_task[task.task_id]
        source_image = args.prepared_dir / metadata.filename
        if not source_image.exists():
            raise FileNotFoundError(source_image)

        image_destination = args.output_dir / "images" / split / source_image.name
        label_destination = args.output_dir / "labels" / split / f"{source_image.stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_destination)
        copied += 1

        lines = []
        for polygon in task.polygons:
            line = polygon_to_yolo_line(polygon, label_to_id)
            if line is None:
                continue
            lines.append(line)
            label = str(polygon["value"]["polygonlabels"][0])
            counts[(split, label)] += 1

        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_data_yaml(args.output_dir, labels)
    print(f"Exported Label Studio project {project_id} ({project_title})")
    print(f"YOLO segmentation dataset: {args.output_dir}")
    print(f"Images: {copied}")
    print("Labels:")
    for label, class_id in label_to_id.items():
        print(f"  {class_id}: {label}")
    print("Polygons:")
    for (split, label), count in sorted(counts.items()):
        print(f"  {split:5s} {label:20s} {count:3d}")
    print(f"Config: {args.output_dir / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-studio-db",
        type=Path,
        default=Path("/home/johannes/.local/share/label-studio/label_studio.sqlite3"),
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("segmentation_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("yolo_seg_dataset"))
    parser.add_argument("--project", help="Label Studio project id or title")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be >= 0 and < 1")
    export_yolo_seg(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
