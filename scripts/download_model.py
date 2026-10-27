"""Download the official MediaPipe Face Landmarker model asset.

Usage (from the project root):
    python scripts/download_model.py

The model file (face_landmarker.task) is required by the MediaPipe Tasks
FaceLandmarker API and is NOT bundled with the mediapipe pip package.
Source URL is the official Google MediaPipe model storage used by the
official documentation:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_RELATIVE_PATH = Path("models") / "face_landmarker.task"
MIN_SIZE_BYTES = 1_000_000  # the real model is several MB; anything smaller is wrong


def download(destination: Path, url: str = MODEL_URL) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model ...")
    print(f"  url: {url}")
    tmp_path = destination.with_suffix(".task.part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as out:
        total = 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    if total < MIN_SIZE_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is only {total} bytes; expected >= {MIN_SIZE_BYTES}. "
            "The URL may be wrong or a proxy may have returned an error page."
        )
    digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    tmp_path.replace(destination)
    print(f"  saved: {destination} ({total / 1e6:.1f} MB)")
    print(f"  sha256: {digest}")
    return destination


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    destination = project_root / MODEL_RELATIVE_PATH
    if destination.exists() and destination.stat().st_size >= MIN_SIZE_BYTES:
        print(f"Model already present: {destination}")
        return 0
    try:
        download(destination)
    except Exception as exc:  # noqa: BLE001 - report any network/IO failure clearly
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Download the file manually from:\n"
            f"  {MODEL_URL}\n"
            f"and place it at: {destination}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
