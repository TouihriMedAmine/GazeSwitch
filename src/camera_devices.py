"""Camera discovery by name.

OpenCV only knows numeric indices, and those shift when devices are plugged
in or removed, so the chosen camera is stored by its device name and
resolved to an (index, backend) pair at open time. Media Foundation devices
come first (faster with high-resolution virtual cameras); DirectShow-only
devices such as OBS Virtual Camera are appended.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDevice:
    name: str
    index: int  # index within its backend
    backend: str  # "msmf" | "dshow"


def list_cameras() -> List[CameraDevice]:
    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        log.warning("cv2_enumerate_cameras not installed - camera names unavailable")
        return []

    devices: List[CameraDevice] = []
    seen = set()
    for backend, label in ((cv2.CAP_MSMF, "msmf"), (cv2.CAP_DSHOW, "dshow")):
        try:
            found = list(enumerate_cameras(backend))
        except Exception as exc:  # noqa: BLE001 - enumeration is best effort
            log.warning("Camera enumeration (%s) failed: %s", label, exc)
            continue
        for cam in found:
            if not cam.name or cam.name in seen:
                continue
            seen.add(cam.name)
            devices.append(CameraDevice(cam.name, int(cam.index), label))
    return devices


def find_camera(name: str, devices: Optional[List[CameraDevice]] = None) -> Optional[CameraDevice]:
    devices = list_cameras() if devices is None else devices
    for device in devices:
        if device.name == name:
            return device
    lowered = name.lower()
    for device in devices:
        if lowered in device.name.lower():
            return device
    return None
