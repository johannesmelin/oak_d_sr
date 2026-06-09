#!/usr/bin/env python3
"""Estimate X/Y/Z from YOLO segmentation masks and OAK-D SR stereo depth."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import depthai as dai
import numpy as np

try:
    from oakdsr_common import open_oakdsr_pipeline, socket_name
except ModuleNotFoundError:
    from scripts.oakdsr_common import open_oakdsr_pipeline, socket_name


DEFAULT_MODEL = Path(
    "/home/johannes/projects/oak_camera/training_runs/"
    "knopp_yolo_seg_640_e30/weights/best.pt"
)
HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]
CAM_TO_GRID_R = np.array(
    [
        [0.999739296, -0.017369757, 0.014819986],
        [-0.022243652, -0.594407549, 0.803856260],
        [-0.005153677, -0.803976342, -0.594638951],
    ],
    dtype=np.float64,
)
CAM_TO_GRID_T = np.array([10.975328, -4.674079, 317.735563], dtype=np.float64)

VIEWER_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OAK-D SR YOLO Localizer</title>
  <style>
    html, body {
      margin: 0;
      min-height: 100%;
      background: #101417;
      color: #eef3f5;
      font-family: system-ui, sans-serif;
    }
    header {
      height: 44px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 14px;
      background: #192126;
      border-bottom: 1px solid #2b363d;
      font-size: 15px;
      font-weight: 650;
    }
    header span {
      color: #b8c8cf;
      font-size: 13px;
      font-weight: 500;
    }
    main {
      min-height: calc(100vh - 45px);
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    img {
      display: block;
      max-width: 100vw;
      max-height: calc(100vh - 45px);
      object-fit: contain;
    }
  </style>
</head>
<body>
  <header>
    <div>OAK-D SR YOLO Localizer</div>
    <span id="status">starting</span>
  </header>
  <main><img src="/stream" alt="OAK-D SR YOLO stream"></main>
  <script>
    async function pollStatus() {
      try {
        const response = await fetch("/status", {cache: "no-store"});
        const data = await response.json();
        document.getElementById("status").textContent =
          `${data.frames} frames, ${data.detections} detections, ${data.socket}`;
      } catch (_error) {
        document.getElementById("status").textContent = "waiting";
      }
    }
    setInterval(pollStatus, 1000);
    pollStatus();
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class SegSpatialDetection:
    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    polygon: np.ndarray | None
    center: tuple[int, int]
    xyz_mm: tuple[float, float, float] | None
    grid_xyz_mm: tuple[float, float, float] | None
    source: str
    support_pixels: int
    depth_z_mm: float | None


@dataclass(frozen=True)
class PositionEstimate:
    xyz_mm: tuple[float, float, float]
    pixel: tuple[int, int]
    source: str
    support_pixels: int
    z_mm: float


class PositionSmoother:
    def __init__(self, window: int, max_jump_mm: float) -> None:
        self.window = max(1, window)
        self.max_jump_mm = max(0.0, max_jump_mm)
        self.history: dict[int, deque[tuple[float, float, float]]] = {}

    def update(self, detections: list[SegSpatialDetection]) -> list[SegSpatialDetection]:
        if self.window <= 1:
            return detections

        smoothed: list[SegSpatialDetection] = []
        active_keys = set()
        for index, detection in enumerate(detections):
            if detection.xyz_mm is None:
                smoothed.append(detection)
                continue

            active_keys.add(index)
            history = self.history.setdefault(index, deque(maxlen=self.window))
            xyz = detection.xyz_mm
            if history and self.max_jump_mm > 0:
                current = np.asarray(xyz, dtype=np.float64)
                previous = np.median(np.asarray(history, dtype=np.float64), axis=0)
                if float(np.linalg.norm(current - previous)) > self.max_jump_mm:
                    xyz = tuple(float(value) for value in previous)

            history.append(xyz)
            median = np.median(np.asarray(history, dtype=np.float64), axis=0)
            smoothed.append(replace(detection, xyz_mm=tuple(float(value) for value in median)))

        for key in list(self.history):
            if key not in active_keys:
                self.history.pop(key)

        return smoothed


class MjpegViewer:
    def __init__(self, host: str, port: int, jpeg_quality: int) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.running = False
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.frames = 0
        self.detections = 0
        self.socket = "unknown"

    def start(self) -> None:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(VIEWER_PAGE.encode("utf-8"))
                    return

                if parsed.path == "/status":
                    with viewer.condition:
                        body = json.dumps(
                            {
                                "running": viewer.running,
                                "frames": viewer.frames,
                                "detections": viewer.detections,
                                "socket": viewer.socket,
                            }
                        ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()

                    while viewer.running:
                        with viewer.condition:
                            viewer.condition.wait(timeout=1.0)
                            jpeg = viewer.jpeg
                        if jpeg is None:
                            continue
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    return

                self.send_response(404)
                self.end_headers()

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, frame: np.ndarray, detections: int, socket: str) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.frames += 1
            self.detections = detections
            self.socket = socket
            self.condition.notify_all()

    def stop(self) -> None:
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"


class RateLimiter:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(0.0, interval_s)
        self.last_time = 0.0

    def ready(self) -> bool:
        if self.interval_s <= 0:
            return True
        now = time.monotonic()
        if now - self.last_time < self.interval_s:
            return False
        self.last_time = now
        return True


def writable_cache_dir(preferred: Path, fallback: Path) -> str:
    for path in (preferred, fallback):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return str(path.resolve())
        except OSError:
            continue
    return str(fallback)


def configure_runtime_cache() -> None:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        writable_cache_dir(Path(".cache/ultralytics"), Path("/tmp/oakdsr_ultralytics")),
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        writable_cache_dir(Path(".cache/matplotlib"), Path("/tmp/oakdsr_matplotlib")),
    )


def create_stereo_queues(
    pipeline: dai.Pipeline,
    width: int,
    height: int,
    fps: int,
    stereo_width: int,
    stereo_height: int,
) -> tuple[dai.MessageQueue, dai.MessageQueue, dai.CameraBoardSocket]:
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    left = pipeline.create(dai.node.Camera).build(left_socket)
    right = pipeline.create(dai.node.Camera).build(right_socket)
    stereo = pipeline.create(dai.node.StereoDepth)

    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(left_socket)
    stereo.setOutputSize(width, height)

    preview_output = left.requestOutput(
        size=(width, height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=fps,
    )
    left.requestOutput((stereo_width, stereo_height), fps=fps).link(stereo.left)
    right.requestOutput((stereo_width, stereo_height), fps=fps).link(stereo.right)

    try:
        rgb_queue = preview_output.createOutputQueue(maxSize=1, blocking=False)
        depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    except TypeError:
        rgb_queue = preview_output.createOutputQueue()
        depth_queue = stereo.depth.createOutputQueue()
    return rgb_queue, depth_queue, left_socket


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 0:
        return frame
    raise ValueError(f"Unsupported camera rotation: {rotation}")


def rotated_intrinsics(intrinsics: np.ndarray, width: int, height: int, rotation: int) -> np.ndarray:
    output = intrinsics.copy()
    if rotation == 180:
        output[0, 2] = (width - 1) - output[0, 2]
        output[1, 2] = (height - 1) - output[1, 2]
    elif rotation != 0:
        raise ValueError(f"Unsupported camera rotation: {rotation}")
    return output


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    width = max(1, int(round(frame.shape[1] * scale)))
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def parse_classes(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_hsv(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be formatted as H,S,V")
    h, s, v = parts
    if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
        raise argparse.ArgumentTypeError("HSV limits are H=0..179, S=0..255, V=0..255")
    return h, s, v


def hsv_config_from_file(path: Path) -> tuple[list[HsvRange], int | None, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    open_kernel = payload.get("open_kernel")
    close_kernel = payload.get("close_kernel")
    if "ranges" in payload:
        return (
            [(tuple(item["low"]), tuple(item["high"])) for item in payload["ranges"]],
            None if open_kernel is None else int(open_kernel),
            None if close_kernel is None else int(close_kernel),
        )

    h_low = int(payload["h_low"])
    h_high = int(payload["h_high"])
    s_low = int(payload["s_low"])
    s_high = int(payload["s_high"])
    v_low = int(payload["v_low"])
    v_high = int(payload["v_high"])
    if bool(payload.get("wrap_hue")):
        ranges = [
            ((h_low, s_low, v_low), (179, s_high, v_high)),
            ((0, s_low, v_low), (h_high, s_high, v_high)),
        ]
    else:
        ranges = [((h_low, s_low, v_low), (h_high, s_high, v_high))]
    return (
        ranges,
        None if open_kernel is None else int(open_kernel),
        None if close_kernel is None else int(close_kernel),
    )


def resolve_hsv_ranges(args: argparse.Namespace) -> list[HsvRange]:
    ranges: list[HsvRange] = []
    if args.hsv_config:
        config_ranges, open_kernel, close_kernel = hsv_config_from_file(args.hsv_config)
        ranges.extend(config_ranges)
        if open_kernel is not None:
            args.hsv_open_kernel = open_kernel
        if close_kernel is not None:
            args.hsv_close_kernel = close_kernel
    if args.hsv_low or args.hsv_high:
        if not (args.hsv_low and args.hsv_high):
            raise ValueError("Use both --hsv-low and --hsv-high.")
        ranges.append((args.hsv_low, args.hsv_high))
    return ranges


def odd_kernel_size(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def hsv_mask(
    frame: np.ndarray,
    ranges: list[HsvRange],
    open_kernel: int,
    close_kernel: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))

    open_size = odd_kernel_size(open_kernel)
    if open_size:
        kernel = np.ones((open_size, open_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_size = odd_kernel_size(close_kernel)
    if close_size:
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def class_label(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def xyz_from_pixel(
    pixel: tuple[int, int],
    z_mm: float,
    intrinsics: np.ndarray,
) -> tuple[float, float, float]:
    u, v = pixel
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_mm = (u - cx) * z_mm / fx
    y_mm = (v - cy) * z_mm / fy
    return x_mm, y_mm, z_mm


def cam_to_grid(xyz_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    grid = CAM_TO_GRID_R @ np.asarray(xyz_mm, dtype=np.float64) + CAM_TO_GRID_T
    return tuple(float(value) for value in grid)


def apply_grid_transform(
    detections: list[SegSpatialDetection],
    enabled: bool,
) -> list[SegSpatialDetection]:
    if not enabled:
        return detections

    transformed = []
    for detection in detections:
        if detection.xyz_mm is None:
            transformed.append(replace(detection, grid_xyz_mm=None))
        else:
            transformed.append(replace(detection, grid_xyz_mm=cam_to_grid(detection.xyz_mm)))
    return transformed


def scaled_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return points.astype(np.int32)
    center = points.mean(axis=0, keepdims=True)
    return np.round(center + (points - center) * scale).astype(np.int32)


def polygon_mask(
    shape: tuple[int, int],
    points: np.ndarray,
    scale: float,
    erode_kernel: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    scaled = scaled_polygon(points.astype(np.float32), scale)
    cv2.fillPoly(mask, [scaled], 255)
    if erode_kernel > 1:
        size = erode_kernel if erode_kernel % 2 == 1 else erode_kernel + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)
    return mask


def depth_estimate_for_mask(
    frame: np.ndarray,
    depth_frame: np.ndarray,
    points: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> PositionEstimate | None:
    mask = polygon_mask(depth_frame.shape[:2], points, args.mask_scale, args.mask_erode)
    source = "seg-mask-depth"
    if args.depth_mask == "hsv":
        color_mask = hsv_mask(frame, args.hsv_ranges, args.hsv_open_kernel, args.hsv_close_kernel)
        hsv_masked = cv2.bitwise_and(mask, color_mask)
        if np.count_nonzero(hsv_masked) >= args.min_depth_pixels:
            mask = hsv_masked
            source = "seg+hsv-depth"
        elif args.hsv_fallback:
            source = "seg-depth-fallback"
        else:
            return None

    valid = (
        (mask > 0)
        & (depth_frame >= args.lower_mm)
        & (depth_frame <= args.upper_mm)
        & np.isfinite(depth_frame)
    )
    values = depth_frame[valid]
    if values.size < args.min_depth_pixels and args.depth_mask == "hsv" and args.hsv_fallback:
        mask = polygon_mask(depth_frame.shape[:2], points, args.mask_scale, args.mask_erode)
        source = "seg-depth-fallback"
        valid = (
            (mask > 0)
            & (depth_frame >= args.lower_mm)
            & (depth_frame <= args.upper_mm)
            & np.isfinite(depth_frame)
        )
        values = depth_frame[valid]
    if values.size < args.min_depth_pixels:
        return None

    z_threshold = float(np.percentile(values, args.depth_percentile))
    foreground = valid & (depth_frame <= z_threshold + args.depth_band_mm)
    if np.count_nonzero(foreground) < args.min_depth_pixels:
        foreground = valid

    ys, xs = np.nonzero(foreground)
    selected_values = depth_frame[foreground]
    z_mm = float(np.median(selected_values))
    pixel = (
        int(round(float(np.median(xs)))),
        int(round(float(np.median(ys)))),
    )
    return PositionEstimate(
        xyz_mm=xyz_from_pixel(pixel, z_mm, intrinsics),
        pixel=pixel,
        source=source,
        support_pixels=int(selected_values.size),
        z_mm=z_mm,
    )


def bbox_from_xyxy(box: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        int(round(max(0, x0))),
        int(round(max(0, y0))),
        int(round(min(frame_width - 1, x1))),
        int(round(min(frame_height - 1, y1))),
    )


def center_from_polygon_or_bbox(
    polygon: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    if polygon is not None and polygon.size:
        center = polygon.mean(axis=0)
        return int(round(center[0])), int(round(center[1]))
    x0, y0, x1, y1 = bbox
    return int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))


def run_seg_localizer(
    model: object,
    frame: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> list[SegSpatialDetection]:
    result = model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.inference_device,
        max_det=args.max_objects,
        verbose=False,
        retina_masks=args.retina_masks,
    )[0]

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)

    detections = []
    for box, confidence, class_id, polygon in zip(boxes, confidences, class_ids, polygons):
        label = class_label(result.names, int(class_id))
        if args.classes and label not in args.classes and str(class_id) not in args.classes:
            continue

        bbox = bbox_from_xyxy(box, frame.shape[1], frame.shape[0])
        points = None
        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon, dtype=np.float32)

        depth = None if points is None else depth_estimate_for_mask(frame, depth_frame, points, intrinsics, args)
        if args.require_position and depth is None:
            continue

        center = center_from_polygon_or_bbox(points, bbox) if depth is None else depth.pixel
        detections.append(
            SegSpatialDetection(
                label=label,
                class_id=int(class_id),
                confidence=float(confidence),
                bbox=bbox,
                polygon=None if points is None else np.round(points).astype(np.int32),
                center=center,
                xyz_mm=None if depth is None else depth.xyz_mm,
                grid_xyz_mm=None,
                source="none" if depth is None else depth.source,
                support_pixels=0 if depth is None else depth.support_pixels,
                depth_z_mm=None if depth is None else depth.z_mm,
            )
        )

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections


def draw_label_box(
    frame: np.ndarray,
    anchor: tuple[int, int],
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_height = 18
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_width = max(size[0] for size in sizes)
    box_width = min(frame.shape[1], text_width + 10)
    box_height = line_height * len(lines) + 8

    x, y = anchor
    x = min(max(0, x), max(0, frame.shape[1] - box_width))
    y = max(box_height + 2, y)
    if y > frame.shape[0] - 2:
        y = frame.shape[0] - 2
    top = max(0, y - box_height)
    bottom = min(frame.shape[0], y + 2)
    right = min(frame.shape[1], x + box_width)

    cv2.rectangle(frame, (x, top), (right, bottom), (15, 20, 24), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 5, top + 17 + line_height * index),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_number_marker(frame: np.ndarray, center: tuple[int, int], index: int) -> None:
    label = str(index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    padding_x = 8
    padding_y = 5
    width = text_size[0] + padding_x * 2
    height = text_size[1] + baseline + padding_y * 2

    x = int(center[0] + 8)
    y = int(center[1] - height - 8)
    x = min(max(0, x), max(0, frame.shape[1] - width))
    y = min(max(34, y), max(34, frame.shape[0] - height))

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 220), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (x + padding_x, y + padding_y + text_size[1]),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_coordinate_list(frame: np.ndarray, detections: list[SegSpatialDetection]) -> None:
    if not detections:
        return

    lines = ["coordinates"]
    for index, detection in enumerate(detections, start=1):
        if detection.xyz_mm is None:
            lines.append(f"{index}: no cam position")
        else:
            x_mm, y_mm, z_mm = detection.xyz_mm
            lines.append(f"{index} cam  x={x_mm:.0f} y={y_mm:.0f} z={z_mm:.0f} mm")

        if detection.grid_xyz_mm is None:
            lines.append("  grid no position")
        else:
            gx_mm, gy_mm, gz_mm = detection.grid_xyz_mm
            lines.append(f"  grid x={gx_mm:.0f} y={gy_mm:.0f} z={gz_mm:.0f} mm")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 22
    padding = 10
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = min(frame.shape[1], max(size[0] for size in text_sizes) + padding * 2)
    height = min(frame.shape[0] - 34, line_height * len(lines) + padding * 2)
    x0, y0 = 0, 34

    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (12, 18, 22), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (55, 70, 78), 1)

    for row, line in enumerate(lines):
        color = (238, 243, 245) if row == 0 else (190, 245, 255)
        cv2.putText(
            frame,
            line,
            (x0 + padding, y0 + padding + 16 + line_height * row),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_detections(frame: np.ndarray, detections: list[SegSpatialDetection], show_boxes: bool) -> None:
    overlay = frame.copy()
    for index, detection in enumerate(detections, start=1):
        color = (0, 255, 255) if detection.xyz_mm is not None else (0, 180, 255)
        x0, y0, x1, y1 = detection.bbox

        if detection.polygon is not None:
            cv2.fillPoly(overlay, [detection.polygon], color)

    if detections:
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for index, detection in enumerate(detections, start=1):
        color = (0, 255, 255) if detection.xyz_mm is not None else (0, 180, 255)
        x0, y0, x1, y1 = detection.bbox

        if detection.polygon is not None:
            cv2.polylines(frame, [detection.polygon], isClosed=True, color=color, thickness=2)
        if show_boxes:
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

        cv2.circle(frame, detection.center, 4, (0, 0, 255), -1)
        draw_number_marker(frame, detection.center, index)

    draw_coordinate_list(frame, detections)


def draw_header(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (12, 18, 22), -1)
    cv2.putText(
        frame,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (238, 243, 245),
        1,
        cv2.LINE_AA,
    )


def print_detections(frame_number: int, detections: list[SegSpatialDetection]) -> None:
    if not detections:
        print(f"{frame_number:05d}  no segmentation detections")
        return

    parts = []
    for index, detection in enumerate(detections, start=1):
        if detection.xyz_mm is None:
            parts.append(f"#{index} {detection.label} conf={detection.confidence:.2f} no-position")
            continue
        x_mm, y_mm, z_mm = detection.xyz_mm
        text = (
            f"#{index} {detection.label} conf={detection.confidence:.2f} "
            f"cam=({x_mm:7.1f},{y_mm:7.1f},{z_mm:7.1f}) "
        )
        if detection.grid_xyz_mm is not None:
            gx_mm, gy_mm, gz_mm = detection.grid_xyz_mm
            text += f"grid=({gx_mm:7.1f},{gy_mm:7.1f},{gz_mm:7.1f}) "
        text += (
            f"source={detection.source} pixels={detection.support_pixels}"
        )
        parts.append(text)
    print(f"{frame_number:05d}  " + "  ".join(parts), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device MXID")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classes", default="knopp", help="Comma-separated class names or ids to keep")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--stereo-width", type=int, default=1280)
    parser.add_argument("--stereo-height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--camera-rotation", type=int, choices=(0, 180), default=180)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--lower-mm", type=int, default=150)
    parser.add_argument("--upper-mm", type=int, default=1200)
    parser.add_argument("--min-depth-pixels", type=int, default=1)
    parser.add_argument("--depth-percentile", type=float, default=20.0)
    parser.add_argument("--depth-band-mm", type=float, default=30.0)
    parser.add_argument(
        "--depth-mask",
        choices=("segmentation", "hsv"),
        default="segmentation",
        help="Use only YOLO segmentation pixels, or YOLO pixels that also match HSV.",
    )
    parser.add_argument("--hsv-config", type=Path)
    parser.add_argument("--hsv-low", type=parse_hsv, help="HSV lower bound as H,S,V")
    parser.add_argument("--hsv-high", type=parse_hsv, help="HSV upper bound as H,S,V")
    parser.add_argument("--hsv-open-kernel", type=int, default=0)
    parser.add_argument("--hsv-close-kernel", type=int, default=2)
    parser.add_argument(
        "--no-hsv-fallback",
        dest="hsv_fallback",
        action="store_false",
        help="Do not fall back to the full segmentation mask if HSV gives too few depth pixels.",
    )
    parser.set_defaults(hsv_fallback=True)
    parser.add_argument("--mask-scale", type=float, default=0.8)
    parser.add_argument("--mask-erode", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--max-jump-mm", type=float, default=0.0)
    parser.add_argument("--require-position", action="store_true")
    parser.add_argument("--retina-masks", action="store_true")
    parser.add_argument("--samples", type=int, default=0, help="0 means run until q/Ctrl+C")
    parser.add_argument("--print-every", type=float, default=0.5)
    parser.add_argument("--viewer-interval-ms", type=int, default=1000)
    parser.add_argument("--viewer-scale", type=float, default=0.75)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8094)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--show-boxes", action="store_true")
    parser.add_argument(
        "--no-grid-transform",
        dest="grid_transform",
        action="store_false",
        help="Hide calibrated grid coordinates and show only camera coordinates.",
    )
    parser.set_defaults(grid_transform=True)
    return parser.parse_args()


def terminal_commands(stop_event: threading.Event) -> None:
    print("Terminal command: q + Enter quits.")
    while not stop_event.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip().lower() == "q":
            stop_event.set()
            return


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Missing model weights: {args.model}")
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be positive")
    if args.stereo_width < 1 or args.stereo_height < 1:
        raise SystemExit("--stereo-width and --stereo-height must be positive")
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    if args.min_depth_pixels < 1:
        raise SystemExit("--min-depth-pixels must be at least 1")
    if not 0 <= args.depth_percentile <= 100:
        raise SystemExit("--depth-percentile must be between 0 and 100")
    if args.depth_band_mm < 0:
        raise SystemExit("--depth-band-mm must be >= 0")
    try:
        args.hsv_ranges = resolve_hsv_ranges(args)
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"Could not read HSV settings: {exc}") from exc
    if args.depth_mask == "hsv" and not args.hsv_ranges:
        raise SystemExit("Use --hsv-config or --hsv-low/--hsv-high with --depth-mask hsv")
    if args.hsv_open_kernel < 0 or args.hsv_close_kernel < 0:
        raise SystemExit("--hsv-open-kernel and --hsv-close-kernel must be >= 0")
    if not 0 < args.mask_scale <= 1:
        raise SystemExit("--mask-scale must be > 0 and <= 1")
    if args.mask_erode < 0:
        raise SystemExit("--mask-erode must be >= 0")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1")
    if args.max_jump_mm < 0:
        raise SystemExit("--max-jump-mm must be >= 0")
    if args.viewer_interval_ms < 0:
        raise SystemExit("--viewer-interval-ms must be >= 0")
    if args.viewer_scale <= 0:
        raise SystemExit("--viewer-scale must be > 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    args.classes = parse_classes(args.classes)
    configure_runtime_cache()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    smoother = PositionSmoother(args.smooth_window, args.max_jump_mm)
    viewer = None
    if not args.no_viewer:
        viewer = MjpegViewer(args.web_host, args.web_port, args.jpeg_quality)
        viewer.start()

    stop_event = threading.Event()
    command_thread = threading.Thread(target=terminal_commands, args=(stop_event,), daemon=True)
    command_thread.start()

    try:
        with open_oakdsr_pipeline(args.device) as pipeline:
            rgb_queue, depth_queue, left_socket = create_stereo_queues(
                pipeline,
                args.width,
                args.height,
                args.fps,
                args.stereo_width,
                args.stereo_height,
            )
            device = pipeline.getDefaultDevice()
            calibration = device.readCalibration()
            intrinsics = np.asarray(
                calibration.getCameraIntrinsics(
                    left_socket,
                    args.width,
                    args.height,
                    keepAspectRatio=True,
                ),
                dtype=np.float64,
            )
            intrinsics = rotated_intrinsics(intrinsics, args.width, args.height, args.camera_rotation)
            left_label = socket_name(left_socket)

            pipeline.start()
            print(f"Opened {device.getDeviceId()}")
            print(f"Product: {device.getProductName()}")
            print(f"Model: {args.model}")
            print(f"Classes: {model.names}")
            print(f"Stereo: CAM_B -> CAM_C, depth aligned to {left_label}")
            print(f"Image: {args.width}x{args.height} at {args.fps} FPS, rotation={args.camera_rotation}")
            print(f"Depth range: {args.lower_mm}-{args.upper_mm} mm")
            print(f"Depth mask: {args.depth_mask}")
            if args.depth_mask == "hsv":
                print(f"Depth HSV ranges: {args.hsv_ranges}")
                print(
                    f"Depth HSV morphology: open={args.hsv_open_kernel}, "
                    f"close={args.hsv_close_kernel}, fallback={args.hsv_fallback}"
                )
            if viewer is not None:
                print(f"Open viewer: {viewer.url}")

            frame_number = 0
            last_print = 0.0
            viewer_limiter = RateLimiter(args.viewer_interval_ms / 1000.0)
            while pipeline.isRunning() and not stop_event.is_set():
                frame = rotate_frame(rgb_queue.get().getCvFrame(), args.camera_rotation)
                depth_frame = rotate_frame(depth_queue.get().getFrame(), args.camera_rotation)

                frame_number += 1
                detections = run_seg_localizer(model, frame, depth_frame, intrinsics, args)
                detections = smoother.update(detections)
                detections = apply_grid_transform(detections, args.grid_transform)

                now = time.monotonic()
                if now - last_print >= args.print_every:
                    print_detections(frame_number, detections)
                    last_print = now

                if viewer is not None and viewer_limiter.ready():
                    view = frame.copy()
                    draw_detections(view, detections, args.show_boxes)
                    draw_header(
                        view,
                        f"OAK-D SR YOLO seg  {left_label}  {args.width}x{args.height}  frame={frame_number}",
                    )
                    view = scale_frame(view, args.viewer_scale)
                    viewer.update(view, len(detections), left_label)

                if args.samples and frame_number >= args.samples:
                    break
    finally:
        stop_event.set()
        if viewer is not None:
            viewer.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
