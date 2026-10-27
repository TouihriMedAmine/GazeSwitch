"""Global shortcuts and the preview's camera picker (no camera, no Win32)."""
from __future__ import annotations

import numpy as np

from main import HOTKEY_ACTIONS
from src.config import HotkeysConfig
from src.input_hotkeys import format_combo, parse_combo
from src.overlay import PreviewRenderer, PreviewView


def test_every_action_has_a_parsable_default_combo():
    cfg = HotkeysConfig()
    combos = [getattr(cfg, name) for name in HOTKEY_ACTIONS]
    assert all(parse_combo(c) is not None for c in combos)
    assert len(set(combos)) == len(combos)  # no two actions share a shortcut


def test_format_combo():
    assert format_combo("ctrl+alt+p") == "Ctrl+Alt+P"
    assert format_combo("shift+win+f5") == "Shift+Win+F5"


def _view(**kw) -> PreviewView:
    return PreviewView(calibrated=True, face=True, camera_name="Iriun Webcam",
                       cameras=("Iriun Webcam", "DroidCam Video"), **kw)


def _center(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def test_camera_pill_and_menu_are_clickable():
    renderer = PreviewRenderer(720)
    frame = np.zeros((480, 640, 3), np.uint8)

    renderer.render(frame, _view())
    pill = dict(renderer._hits)["camera_menu"]
    assert renderer.hit_test(*_center(pill)) == "camera_menu"
    assert "camera:DroidCam Video" not in dict(renderer._hits)  # menu closed

    renderer.render(frame, _view(camera_menu=True))
    hits = dict(renderer._hits)
    for name in ("camera:", "camera:Iriun Webcam", "camera:DroidCam Video", "refresh_cameras"):
        assert renderer.hit_test(*_center(hits[name])) == name
    assert renderer.hit_test(700, 500) is None  # outside: closes the menu
