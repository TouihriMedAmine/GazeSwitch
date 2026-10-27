"""Preview UI tests: copy for each state and a render smoke test."""
from __future__ import annotations

import numpy as np

from src.monitor_classifier import MonitorState
from src.overlay import PreviewRenderer, PreviewView, headline, window_label


def test_window_label_drops_process_name():
    assert window_label("Cursor.exe - main.py - Gaze_Tracker - Cursor") == \
        "main.py - Gaze_Tracker - Cursor"
    assert window_label("none") == ""
    assert window_label("notepad.exe") == "notepad"


def test_headline_per_state():
    assert headline(PreviewView(camera_ok=False))[0] == "Waiting for the camera"
    assert headline(PreviewView(face=False, calibrated=True))[0] == "Can't see your face"
    assert headline(PreviewView(face=True, calibrated=False))[0] == "Calibrate to start"
    assert headline(PreviewView(face=True, calibrated=True))[0] == "Looking away"
    title, sub = headline(PreviewView(face=True, calibrated=True, state=MonitorState.LEFT,
                                      target="brave.exe - YouTube - Brave"))
    assert (title, sub) == ("Left monitor", "YouTube - Brave")
    assert headline(PreviewView(face=True, calibrated=True, state=MonitorState.RIGHT,
                                target="none"))[1] == "No window on this monitor"


def test_render_keeps_aspect_and_changes_pixels():
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    renderer = PreviewRenderer(720)
    landmarks = np.full((478, 3), 0.5, dtype=np.float32)
    for view in (
        PreviewView(face=True, calibrated=True, state=MonitorState.CENTER, stable=True,
                    landmarks=landmarks, message="Calibrated", details=True,
                    detail_lines=["x +0.000"]),
        PreviewView(calibrating=True, calib_progress=0.5, calib_valid=20, calib_needed=40),
        PreviewView(camera_ok=False),
    ):
        out = renderer.render(frame, view)
        assert out.shape == (405, 720, 3)
        assert out.dtype == np.uint8
        assert not np.all(out == out[0, 0])
