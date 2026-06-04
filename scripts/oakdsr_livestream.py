#!/usr/bin/env python3
"""Show an OAK-D SR camera stream in a browser."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

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
  <title>OAK-D SR Livestream</title>
  <style>
    html, body {
      margin: 0;
      min-height: 100%;
      background: #111416;
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
      background: #1a2227;
      border-bottom: 1px solid #2b363d;
      font-size: 15px;
      font-weight: 650;
    }
    header span {
      color: #b8c8cf;
      font-weight: 500;
      font-size: 13px;
    }
    main {
      display: grid;
      place-items: center;
      min-height: calc(100vh - 45px);
      overflow: hidden;
    }
    img {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 45px);
      object-fit: contain;
    }
  </style>
</head>
<body>
  <header>
    <div>OAK-D SR Livestream</div>
    <span id="status">starting</span>
  </header>
  <main><img src="/stream" alt="OAK-D SR stream"></main>
  <script>
    async function pollStatus() {
      try {
        const response = await fetch("/status", {cache: "no-store"});
        const data = await response.json();
        document.getElementById("status").textContent =
          `${data.socket} ${data.width}x${data.height} frame ${data.frames}`;
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


class SharedState:
    def __init__(self, jpeg_quality: int) -> None:
        self.jpeg_quality = jpeg_quality
        self.jpeg: bytes | None = None
        self.running = True
        self.frames = 0
        self.socket = "unknown"
        self.width = 0
        self.height = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def update_metadata(self, socket: str, width: int, height: int) -> None:
        with self.lock:
            self.socket = socket
            self.width = width
            self.height = height

    def update_frame(self, frame: np.ndarray) -> None:
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
            self.condition.notify_all()

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "frames": self.frames,
                "socket": self.socket,
                "width": self.width,
                "height": self.height,
            }

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()


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


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 0:
        return frame
    raise ValueError(f"Unsupported camera rotation: {rotation}")


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    width = max(1, int(round(frame.shape[1] * scale)))
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_overlay(frame: np.ndarray, socket: str, frame_count: int) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (12, 18, 22), -1)
    text = f"OAK-D SR  socket={socket}  {output.shape[1]}x{output.shape[0]}  frame={frame_count}"
    cv2.putText(
        output,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (238, 243, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def make_handler(state: SharedState) -> type[BaseHTTPRequestHandler]:
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
                self.wfile.write(PAGE.encode("utf-8"))
                return

            if parsed.path == "/status":
                import json

                body = json.dumps(state.status()).encode("utf-8")
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


def terminal_commands(state: SharedState) -> None:
    print("Terminal command: q + Enter quits.")
    while state.running:
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip().lower() == "q":
            state.stop()
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device MXID")
    parser.add_argument("--socket", choices=("auto", "cam_a", "cam_b", "cam_c"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--camera-rotation", type=int, choices=(0, 180), default=0)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8092)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--viewer-scale", type=float, default=1.0)
    parser.add_argument(
        "--viewer-interval-ms",
        type=int,
        default=0,
        help="Minimum time between browser frames; 0 streams every camera frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be positive")
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    if args.viewer_scale <= 0:
        raise SystemExit("--viewer-scale must be > 0")
    if args.viewer_interval_ms < 0:
        raise SystemExit("--viewer-interval-ms must be >= 0")

    state = SharedState(args.jpeg_quality)
    server = ThreadingHTTPServer((args.web_host, args.web_port), make_handler(state))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    command_thread = threading.Thread(target=terminal_commands, args=(state,), daemon=True)
    command_thread.start()

    host = "127.0.0.1" if args.web_host in ("0.0.0.0", "") else args.web_host
    print(f"Open viewer: http://{host}:{args.web_port}")

    try:
        with open_oakdsr_pipeline(args.device) as pipeline:
            device = pipeline.getDefaultDevice()
            connected = device.getConnectedCameras()
            socket = choose_camera_socket(connected, args.socket)
            queue = create_camera_queue(pipeline, socket, args.width, args.height, args.fps)
            socket_label = socket_name(socket)

            pipeline.start()
            print(f"Opened {device.getDeviceId()}")
            print(f"Product: {device.getProductName()}")
            print(f"Using socket: {socket_label}")
            print(f"Requested image: {args.width}x{args.height} at {args.fps} FPS")
            state.update_metadata(socket_label, args.width, args.height)

            limiter = RateLimiter(args.viewer_interval_ms / 1000.0)
            while pipeline.isRunning() and state.running:
                frame = rotate_frame(queue.get().getCvFrame(), args.camera_rotation)
                if not limiter.ready():
                    continue
                status = state.status()
                view = draw_overlay(frame, socket_label, int(status["frames"]) + 1)
                view = scale_frame(view, args.viewer_scale)
                state.update_metadata(socket_label, view.shape[1], view.shape[0])
                state.update_frame(view)
    finally:
        state.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
        time.sleep(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
