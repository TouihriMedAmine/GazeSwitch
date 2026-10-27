"""Camera tests with injected frame sources: failure handling, reconnection
state transitions, mirroring. No physical camera needed."""
from __future__ import annotations

import numpy as np
import pytest

from src.camera import CameraCapture, CameraStatus


def frame(value=128) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def test_ok_source_produces_frames():
    cap = CameraCapture(frame_source=lambda: (True, frame()), mirror=False)
    assert cap.open() is True
    ok, f = cap.read()
    assert ok and f is not None
    assert cap.status is CameraStatus.OK


def test_mirroring_flips_horizontally():
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[:, 0] = 255  # leftmost column bright
    cap = CameraCapture(frame_source=lambda: (True, img), mirror=True)
    cap.open()
    ok, f = cap.read()
    assert ok
    assert f[:, -1, 0].max() == 255  # bright column moved to the right edge
    assert f[:, 0, 0].max() == 0


def test_failures_do_not_crash_and_status_degrades():
    calls = {"n": 0}

    def source():
        calls["n"] += 1
        return (False, None)

    cap = CameraCapture(frame_source=source, max_read_failures=5)
    cap.open()
    for _ in range(4):
        ok, f = cap.read()
        assert not ok and f is None
        assert cap.status is CameraStatus.RECONNECTING
    cap.read()  # 5th failure
    assert cap.status is CameraStatus.FAILED


def test_recovery_after_failures_restores_ok():
    state = {"fail": True}
    cap = CameraCapture(
        frame_source=lambda: (False, None) if state["fail"] else (True, frame()),
        max_read_failures=3,
    )
    cap.open()
    for _ in range(4):
        cap.read()
    assert cap.status is CameraStatus.FAILED
    state["fail"] = False
    ok, f = cap.read()
    assert ok and f is not None
    assert cap.status is CameraStatus.OK


def test_injected_open_failure_is_reported():
    cap = CameraCapture(index=99, open_fn=lambda idx: False)
    assert cap.open() is False
    assert cap.status is CameraStatus.FAILED


def test_never_raises_on_exception_source():
    def bad_source():
        raise RuntimeError("device vanished")

    cap = CameraCapture(frame_source=bad_source, max_read_failures=2)
    cap.open()
    # read() must swallow the exception source (treated as failed read)
    try:
        ok, f = cap.read()
        assert not ok and f is None
    except RuntimeError:
        pytest.fail("CameraCapture must not propagate source exceptions")
