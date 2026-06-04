"""Shared helpers for the OAK-D SR test lab."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import depthai as dai


SOCKETS: dict[str, dai.CameraBoardSocket] = {
    "cam_a": dai.CameraBoardSocket.CAM_A,
    "cam_b": dai.CameraBoardSocket.CAM_B,
    "cam_c": dai.CameraBoardSocket.CAM_C,
}


@contextmanager
def open_oakdsr_pipeline(device_selector: str | None = None) -> Iterator[dai.Pipeline]:
    """Create a DepthAI v3 pipeline, optionally bound to a specific MXID."""
    device: dai.Device | None = None

    try:
        if device_selector:
            device = dai.Device(dai.DeviceInfo(device_selector))
            with dai.Pipeline(device) as pipeline:
                yield pipeline
        else:
            with dai.Pipeline() as pipeline:
                yield pipeline
    finally:
        if device is not None and not device.isClosed():
            device.close()


def socket_name(socket: dai.CameraBoardSocket) -> str:
    name = getattr(socket, "name", str(socket))
    return name.replace("CameraBoardSocket.", "")


def normalize_connected_sockets(
    sockets: Iterable[dai.CameraBoardSocket],
) -> list[dai.CameraBoardSocket]:
    return list(sockets)


def choose_camera_socket(
    connected: Iterable[dai.CameraBoardSocket],
    selector: str,
) -> dai.CameraBoardSocket:
    connected_list = normalize_connected_sockets(connected)
    if selector != "auto":
        socket = SOCKETS[selector]
        if socket not in connected_list:
            connected_names = ", ".join(socket_name(item) for item in connected_list) or "none"
            raise RuntimeError(
                f"Selected {selector} is not connected. Connected sockets: {connected_names}"
            )
        return socket

    for socket in (
        dai.CameraBoardSocket.CAM_A,
        dai.CameraBoardSocket.CAM_B,
        dai.CameraBoardSocket.CAM_C,
    ):
        if socket in connected_list:
            return socket

    raise RuntimeError("No connected cameras reported by DepthAI")
