#!/usr/bin/env python3
"""Open an OAK-D SR and print basic connection details."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import depthai as dai

try:
    from oakdsr_common import socket_name
except ModuleNotFoundError:
    from scripts.oakdsr_common import socket_name


def describe_device_info(info: dai.DeviceInfo) -> str:
    parts: list[str] = []
    for label, attr in (
        ("mxid", "mxid"),
        ("name", "name"),
        ("state", "state"),
        ("protocol", "protocol"),
    ):
        value = getattr(info, attr, None)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else repr(info)


def print_optional(label: str, getter: Any) -> None:
    try:
        print(f"{label}: {getter()}")
    except Exception as exc:
        print(f"{label}: unavailable ({exc})")


def print_camera_features(device: dai.Device) -> None:
    try:
        features = device.getConnectedCameraFeatures()
    except Exception:
        features = []

    if features:
        print("Connected camera features:")
        for feature in features:
            socket = getattr(feature, "socket", "unknown")
            sensor_name = getattr(feature, "sensorName", "unknown")
            width = getattr(feature, "width", "?")
            height = getattr(feature, "height", "?")
            print(f"  - socket={socket}, sensor={sensor_name}, size={width}x{height}")
        return

    cameras = device.getConnectedCameras()
    print("Connected cameras:")
    for socket in cameras:
        sensor_name = device.getCameraSensorNames().get(socket, "unknown")
        print(f"  - socket={socket_name(socket)}, sensor={sensor_name}")


def print_intrinsics(device: dai.Device) -> None:
    calibration = device.readCalibration()
    for socket in device.getConnectedCameras():
        try:
            intrinsics = calibration.getCameraIntrinsics(
                socket,
                640,
                400,
                keepAspectRatio=True,
            )
        except Exception as exc:
            print(f"{socket_name(socket)} intrinsics at 640x400: unavailable ({exc})")
            continue

        print(f"{socket_name(socket)} intrinsics at 640x400:")
        for row in intrinsics:
            print("  " + " ".join(f"{value:9.3f}" for value in row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device MXID")
    args = parser.parse_args()

    available = dai.Device.getAllAvailableDevices()
    print(f"Discovered devices: {len(available)}")
    for index, info in enumerate(available, start=1):
        print(f"  {index}. {describe_device_info(info)}")
    sys.stdout.flush()

    device_info = dai.DeviceInfo(args.device) if args.device else None
    try:
        if device_info:
            device_context = dai.Device(device_info)
        else:
            device_context = dai.Device()
    except RuntimeError as exc:
        print()
        print(f"Could not open OAK-D SR: {exc}", file=sys.stderr)
        print("Check USB cable, power, udev rules, and that no other process owns the camera.", file=sys.stderr)
        return 1

    with device_context as device:
        print()
        print("Opened device")
        print_optional("Device ID", device.getDeviceId)
        print_optional("Device name", device.getDeviceName)
        print_optional("Product name", device.getProductName)
        print_optional("USB speed", device.getUsbSpeed)
        print_optional("Bootloader version", device.getBootloaderVersion)
        print_camera_features(device)
        print()
        print_intrinsics(device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
