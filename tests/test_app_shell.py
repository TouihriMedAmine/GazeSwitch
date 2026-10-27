"""Desktop-app shell tests: time-based voting, tray state, resource paths.
No camera, tray icon or registry access needed."""
from __future__ import annotations

from src import app_env
from src.monitor_classifier import Classification, MonitorState
from src.selection_state import SelectionConfig, SelectionStateMachine
from src.tray import COLOR_ARMED, COLOR_OBSERVING, COLOR_UNCALIBRATED, TrayController, TrayState


def cls(state: MonitorState) -> Classification:
    return Classification(state=state, confidence=0.8, x=0.0)


def test_time_window_switches_independently_of_frame_count():
    """At 40 fps a 250 ms window holds ~10 frames; switching must still take
    roughly vote majority + dwell, not window_frames worth of frames."""
    cfg = SelectionConfig(window_frames=7, min_confidence=0.6, dwell_ms=200,
                          lost_ms=600, vote_window_ms=250)
    sm = SelectionStateMachine(cfg)
    t = 0
    while t <= 1000:
        sm.update(cls(MonitorState.CENTER), True, t)
        t += 25
    assert sm.selected is MonitorState.CENTER

    start = t
    while sm.selected is MonitorState.CENTER and t < start + 2000:
        sm.update(cls(MonitorState.LEFT), True, t)
        t += 25
    assert sm.selected is MonitorState.LEFT
    latency = t - start
    assert 300 <= latency <= 450  # ~60% of 250 ms + 200 ms dwell


def test_time_window_drops_old_votes():
    cfg = SelectionConfig(min_confidence=0.6, dwell_ms=0, vote_window_ms=100)
    sm = SelectionStateMachine(cfg)
    for t in range(0, 500, 10):
        sm.update(cls(MonitorState.CENTER), True, t)
    assert len(sm.window) <= 11


def test_frame_window_still_default():
    sm = SelectionStateMachine(SelectionConfig(window_frames=5))
    for t in range(0, 2000, 10):
        sm.update(cls(MonitorState.CENTER), True, t)
    assert len(sm.window) == 5


def test_app_icon_is_the_artwork():
    from src.tray import make_icon_image

    icon = make_icon_image(size=64)
    assert icon.size == (64, 64) and icon.mode == "RGBA"
    # The cyan gaze mark from the three-panel artwork is in the icon.
    pixels = zip(*[iter(icon.get_flattened_data())] * 4)
    assert any(r < 80 and g > 180 and b > 220 for r, g, b, _ in pixels)


def test_tray_colors_follow_state():
    assert TrayController._color(TrayState(calibrated=False, armed=True)) == COLOR_UNCALIBRATED
    assert TrayController._color(TrayState(calibrated=True, armed=True)) == COLOR_ARMED
    assert TrayController._color(TrayState(calibrated=True, armed=False)) == COLOR_OBSERVING


def test_tray_queue_and_state_without_icon():
    tray = TrayController()  # never started: no icon, must not raise
    tray._request("calibrate")(None, None)
    tray._request("quit")(None, None)
    assert tray.poll() == "calibrate"
    assert tray.poll() == "quit"
    assert tray.poll() is None
    tray.update_state(monitor="LEFT", armed=True, calibrated=True)
    assert tray._state.monitor == "LEFT"
    assert "switching ON" in TrayController._tooltip(tray._state)
    tray.notify("hello")


def test_resolve_resource_prefers_existing_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.txt").write_text("x")
    assert app_env.resolve_resource("local.txt") == "local.txt"
    # Missing locally and not bundled -> returned unchanged (caller reports it)
    assert app_env.resolve_resource("missing/file.bin") == str(
        app_env.Path("missing/file.bin")
    )
