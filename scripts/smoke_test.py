"""One-frame end-to-end smoke test: camera -> FaceTracker -> features.

Usage:  python scripts/smoke_test.py [camera_index]
Grabs ONE live frame, runs MediaPipe face tracking, prints derived features.
Read-only: no window is opened, nothing is stored or uploaded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    print(f"camera index {index}: ok={ok} frame={None if frame is None else frame.shape} "
          f"negotiated={resolution}")
    if not ok or frame is None:
        print("FAIL: no frame from camera")
        return 1

    from src.face_tracker import FaceTracker

    tracker = FaceTracker(model_path="models/face_landmarker.task")
    t0 = time.perf_counter()
    obs = tracker.detect(frame, int(time.monotonic() * 1000))
    dt_ms = (time.perf_counter() - t0) * 1000
    tracker.close()
    if obs is None:
        print(f"NO FACE in this single frame (inference took {dt_ms:.0f} ms) - "
              "expected if nobody is in the camera's view")
        return 0
    print(
        f"FACE FOUND ({dt_ms:.0f} ms): yaw={obs.yaw:+.3f} pitch={obs.pitch:+.3f} "
        f"iris_dx={obs.iris_dx:+.3f} iris_dy={obs.iris_dy:+.3f} blink={obs.blink:.2f} "
        f"face_w={obs.face_width:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
