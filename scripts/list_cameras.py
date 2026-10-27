"""List cameras by name and check which ones deliver frames.

Run from the project root:

    python main.py --list-cameras
    # or
    python scripts/list_cameras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def list_cameras() -> None:
    import cv2

    from src.camera_devices import list_cameras as discover

    backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
    devices = discover()
    if not devices:
        print("No cameras found. Is your camera (or Camo / Iriun / DroidCam) running?")
        return
    print("Cameras:")
    for device in devices:
        cap = cv2.VideoCapture(device.index, backends[device.backend])
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        status = f"works, {frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None \
            else "no frames (in use by another app, or not streaming)"
        print(f"  {device.name:32s} {device.backend} index {device.index}: {status}")
    print("\nPick one in the tray menu (Camera), or set it in config.yaml:\n"
          f"  camera:\n    name: \"{devices[0].name}\"")


if __name__ == "__main__":
    list_cameras()
