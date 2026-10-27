"""Camera selection and config write-back tests (no camera needed)."""
from __future__ import annotations

import yaml

from main import camera_candidates
from src.camera_devices import CameraDevice, find_camera
from src.config import load_config, set_config_value

DEVICES = [
    CameraDevice("Iriun Webcam", 0, "msmf"),
    CameraDevice("DroidCam Video", 1, "msmf"),
    CameraDevice("OBS Virtual Camera", 2, "dshow"),
]


def test_named_camera_wins():
    assert camera_candidates("DroidCam Video", 0, "auto", DEVICES) == [(1, "msmf", "DroidCam Video")]


def test_partial_name_match():
    assert find_camera("obs", DEVICES).name == "OBS Virtual Camera"


def test_missing_name_falls_back_to_all_devices_in_order():
    result = camera_candidates("Gone Camera", None, "auto", DEVICES)
    assert [label for _, _, label in result] == [d.name for d in DEVICES]


def test_index_used_when_no_name():
    assert camera_candidates(None, 1, "msmf", DEVICES) == [(1, "msmf", "DroidCam Video")]
    assert camera_candidates(None, 5, "auto", DEVICES) == [(5, "auto", "Camera 5")]


def test_no_enumeration_scans_indices():
    assert [i for i, _, _ in camera_candidates(None, None, "auto", [])] == list(range(6))


def test_set_config_value_keeps_comments_and_other_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# top comment\n"
        "camera:\n"
        "  index: null   # keep me\n"
        "  name: null\n"
        "  fps: 30\n"
        "\n"
        "tracking:\n"
        "  blendshapes: false\n",
        encoding="utf-8",
    )
    set_config_value(str(path), "camera", "name", 'Cam "A" \\ 1')
    text = path.read_text(encoding="utf-8")
    assert "# top comment" in text and "# keep me" in text
    data = yaml.safe_load(text)
    assert data["camera"]["name"] == 'Cam "A" \\ 1'
    assert data["camera"]["fps"] == 30
    assert data["tracking"]["blendshapes"] is False

    set_config_value(str(path), "camera", "name", None)
    assert load_config(str(path)).camera.name is None


def test_set_config_value_inserts_missing_key_and_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  fps: 30\n", encoding="utf-8")
    set_config_value(str(path), "camera", "name", "Iriun Webcam")
    set_config_value(str(path), "overlay", "always_on_top", False)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {"camera": {"name": "Iriun Webcam", "fps": 30},
                    "overlay": {"always_on_top": False}}
