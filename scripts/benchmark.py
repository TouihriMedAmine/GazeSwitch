"""Measure where frame time goes: camera delivery rate vs face detection.

    python scripts/benchmark.py [--camera 0] [--seconds 8]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.config import load_config  # noqa: E402
from src.face_tracker import FaceTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "config.yaml"))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
    ok, frame = cap.read()
    if not ok:
        print("camera read failed")
        return 1
    print(f"camera delivers {frame.shape[1]}x{frame.shape[0]}, "
          f"reports fps={cap.get(cv2.CAP_PROP_FPS):.1f}")

    reads, t_end = [], time.monotonic() + args.seconds / 2
    last = time.monotonic()
    frames = []
    while time.monotonic() < t_end:
        ok, frame = cap.read()
        now = time.monotonic()
        reads.append(now - last)
        last = now
        if ok:
            frames.append(frame)
    cap.release()
    print(f"camera only: {1 / statistics.mean(reads):.1f} fps "
          f"(median interval {statistics.median(reads) * 1000:.0f} ms)")

    for blend in (True, False):
        tracker = FaceTracker(model_path=str(ROOT / cfg.tracking.model_path), blendshapes=blend)
        times, faces = [], 0
        t_ms = 0
        for f in frames:
            small = f
            if f.shape[1] > cfg.camera.process_width:
                h = int(f.shape[0] * cfg.camera.process_width / f.shape[1])
                small = cv2.resize(f, (cfg.camera.process_width, h))
            t0 = time.perf_counter()
            obs = tracker.detect(small, t_ms)
            times.append(time.perf_counter() - t0)
            faces += obs is not None
            t_ms += 33
        tracker.close()
        print(f"detection (blendshapes={blend}): median {statistics.median(times) * 1000:.1f} ms"
              f" -> max {1 / statistics.median(times):.0f} fps, face found {faces}/{len(frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
