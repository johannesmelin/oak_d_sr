#!/usr/bin/env python3
"""Capture OAK-D SR images for a segmentation dataset using a browser preview."""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

try:
    from oakdsr_common import choose_camera_socket, open_oakdsr_pipeline, socket_name
except ModuleNotFoundError:
    from scripts.oakdsr_common import choose_camera_socket, open_oakdsr_pipeline, socket_name


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OAK-D SR Dataset Capture</title>
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
      padding: 0 14px;
      background: #192126;
      border-bottom: 1px solid #2b363d;
      font-size: 15px;
      font-weight: 650;
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
  <header>OAK-D SR Dataset Capture</header>
  <main><img src="/stream" alt="OAK-D SR camera stream"></main>
</body>
</html>
"""


@dataclass(frozen=True)
class CaptureMetadata:
    filename: str
    object_class: str
    split: str
    width: int
    height: int
    device_id: str
    camera_socket: str
    timestamp: str
    note: str


class SharedState:
    def __init__(self, jpeg_quality: int) -> None:
        self.jpeg_quality = jpeg_quality
        self.jpeg: bytes | None = None
        self.running = True
        self.saved_count = 0
        self.pending_capture = 0
        self.pending_background = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def update_jpeg(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.condition.notify_all()

    def request_capture(self, count: int = 1) -> None:
        with self.lock:
            self.pending_capture += count

    def request_background(self, count: int = 1) -> None:
        with self.lock:
            self.pending_background += count

    def next_save_class(self, object_class: str) -> str | None:
        with self.lock:
            if self.pending_background > 0:
                self.pending_background -= 1
                return "background"
            if self.pending_capture > 0:
                self.pending_capture -= 1
                return object_class
            return None

    def mark_saved(self) -> int:
        with self.lock:
            self.saved_count += 1
            return self.saved_count

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 0:
        return frame
    raise ValueError(f"Unsupported camera rotation: {rotation}")


def create_camera_queue(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket,
    width: int,
    height: int,
    fps: int,
) -> dai.MessageQueue:
    camera = pipeline.create(dai.node.Camera).build(socket)
    output = camera.requestOutput(
        size=(width, height),
        type=dai.ImgFrame.Type.BGR888p,
        fps=fps,
    )
    try:
        return output.createOutputQueue(maxSize=1, blocking=False)
    except TypeError:
        return output.createOutputQueue()


def sanitize(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "unknown"


def metadata_path(dataset_dir: Path) -> Path:
    return dataset_dir / "metadata.csv"


def append_metadata(dataset_dir: Path, row: CaptureMetadata) -> None:
    path = metadata_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(row)))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def next_filename(object_class: str, count: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{sanitize(object_class)}_{count:05d}.jpg"


def save_capture(
    frame: np.ndarray,
    args: argparse.Namespace,
    device_id: str,
    camera_socket: str,
    object_class: str,
    count: int,
) -> Path:
    filename = next_filename(object_class, count)
    relative_path = Path("images") / args.split / sanitize(object_class) / filename
    output_path = args.dataset_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
        raise RuntimeError(f"Could not write image: {output_path}")

    append_metadata(
        args.dataset_dir,
        CaptureMetadata(
            filename=str(relative_path),
            object_class=object_class,
            split=args.split,
            width=frame.shape[1],
            height=frame.shape[0],
            device_id=device_id,
            camera_socket=camera_socket,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            note=args.note,
        ),
    )
    return output_path


def draw_overlay(frame: np.ndarray, args: argparse.Namespace, count: int, object_class: str, socket: str) -> None:
    lines = [
        f"class: {object_class}",
        f"split: {args.split}  socket: {socket}",
        f"saved: {count}",
        "terminal: c capture | b burst | g background | q quit",
    ]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), (15, 20, 24), -1)
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def make_handler(state: SharedState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(PAGE.encode("utf-8"))
                return

            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while state.running:
                    with state.condition:
                        state.condition.wait(timeout=1.0)
                        jpeg = state.jpeg
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

    return Handler


def terminal_commands(state: SharedState, burst: int) -> None:
    print("Terminal commands: c + Enter captures, b + Enter bursts, g + Enter saves background, q + Enter quits.")
    while state.running:
        line = sys.stdin.readline()
        if not line:
            return
        command = line.strip().lower()
        if command == "c":
            state.request_capture()
        elif command == "b":
            state.request_capture(burst)
        elif command == "g":
            state.request_background()
        elif command == "q":
            state.stop()
            return
        elif command:
            print(f"Unknown command: {command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device MXID")
    parser.add_argument("--class-name", required=True, help="Object class for captured images")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--dataset-dir", type=Path, default=Path("segmentation_dataset"))
    parser.add_argument("--socket", choices=("auto", "cam_b", "cam_c"), default="cam_b")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--camera-rotation", type=int, choices=(0, 180), default=180)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--burst", type=int, default=5, help="Images saved when entering b")
    parser.add_argument("--note", default="")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8095)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    object_class = sanitize(args.class_name)
    state = SharedState(args.jpeg_quality)

    server = ThreadingHTTPServer((args.web_host, args.web_port), make_handler(state))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    command_thread = threading.Thread(target=terminal_commands, args=(state, args.burst), daemon=True)
    command_thread.start()

    host = "127.0.0.1" if args.web_host in ("0.0.0.0", "") else args.web_host
    print(f"Open viewer: http://{host}:{args.web_port}")

    try:
        with open_oakdsr_pipeline(args.device) as pipeline:
            device = pipeline.getDefaultDevice()
            socket = choose_camera_socket(device.getConnectedCameras(), args.socket)
            socket_label = socket_name(socket)
            queue = create_camera_queue(pipeline, socket, args.width, args.height, args.fps)
            device_id = device.getDeviceId()

            pipeline.start()
            print(f"Opened {device_id}")
            print(f"Using socket: {socket_label}")
            print(f"Dataset directory: {args.dataset_dir}")

            while pipeline.isRunning() and state.running:
                frame = rotate_frame(queue.get().getCvFrame(), args.camera_rotation)
                save_class = state.next_save_class(object_class)
                if save_class is not None:
                    count = state.mark_saved()
                    path = save_capture(frame, args, device_id, socket_label, save_class, count)
                    print(f"saved {path}")

                preview = frame.copy()
                draw_overlay(preview, args, state.saved_count, object_class, socket_label)
                state.update_jpeg(preview)
    finally:
        state.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
        time.sleep(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
