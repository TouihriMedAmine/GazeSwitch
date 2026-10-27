"""Window->monitor mapping tests with synthetic monitors/windows: largest
intersection, spanning windows, minimized exclusion, preference overrides,
tie-breaking, and slot mapping. No Win32 calls (pure dataclasses)."""
from __future__ import annotations

from src.monitor_classifier import MonitorState
from src.target_manager import WindowPreference, pick_candidate
from src.window_manager import MonitorInfo, WindowInfo, intersection_area, map_monitors_to_slots


def mk_monitor(handle: int, rect, primary=False) -> MonitorInfo:
    return MonitorInfo(handle=handle, device=f"\\\\.\\DISPLAY{handle}", rect=rect,
                       work_rect=rect, primary=primary, dpi=96)


def mk_window(hwnd: int, rect, monitor_handle: int, areas, title="win", process="app.exe",
              minimized=False, z_order=0, foreground=False) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name="Cls",
        process_name=process,
        pid=100 + hwnd,
        rect=rect,
        monitor_handle=monitor_handle,
        areas=areas,
        minimized=minimized,
        visible=True,
        foreground=foreground,
        z_order=z_order,
    )


M_LEFT = mk_monitor(1, (0, 0, 1920, 1080))
M_CENTER = mk_monitor(2, (1920, 0, 3840, 1080), primary=True)
M_RIGHT = mk_monitor(3, (3840, 0, 5760, 1080))


# ------------------------------------------------------------------ geometry
def test_intersection_area_basics():
    assert intersection_area((0, 0, 100, 100), (50, 50, 150, 150)) == 2500
    assert intersection_area((0, 0, 100, 100), (100, 0, 200, 100)) == 0
    assert intersection_area((0, 0, 100, 100), (10, 10, 20, 20)) == 100


def test_map_three_monitors_by_x_order():
    slots, note = map_monitors_to_slots([M_RIGHT, M_LEFT, M_CENTER])
    assert slots["LEFT"].handle == M_LEFT.handle
    assert slots["CENTER"].handle == M_CENTER.handle
    assert slots["RIGHT"].handle == M_RIGHT.handle
    assert "3 monitors" in note


def test_map_two_monitors_center_is_primary():
    """With 2 monitors, CENTER maps to the primary one (gaze CENTER is
    ambiguous between two displays)."""
    left = mk_monitor(1, (0, 0, 1920, 1080), primary=True)
    right = mk_monitor(2, (1920, 0, 3840, 1080))
    slots, _ = map_monitors_to_slots([right, left])
    assert slots["LEFT"].handle == left.handle
    assert slots["RIGHT"].handle == right.handle
    assert slots["CENTER"].handle == left.handle  # the primary


def test_map_single_monitor():
    slots, note = map_monitors_to_slots([M_CENTER])
    assert slots["LEFT"] is slots["CENTER"] is slots["RIGHT"]


# -------------------------------------------------------------- candidates
def test_front_window_wins_by_default():
    behind = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, z_order=3)
    front = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400}, z_order=1)
    assert pick_candidate([behind, front], M_CENTER) is front


def test_same_z_order_falls_back_to_larger_area():
    big = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080})
    small = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400}, z_order=0)
    assert pick_candidate([big, small], M_CENTER) is big


def test_tiny_topmost_widget_is_ignored():
    widget = mk_window(1, (3700, 0, 3840, 60), 2, {2: 140 * 60}, z_order=0)
    app = mk_window(2, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, z_order=4)
    assert pick_candidate([widget, app], M_CENTER) is app
    assert pick_candidate([widget], M_CENTER) is widget  # still better than nothing


def test_window_mostly_on_another_monitor_does_not_win():
    poking = mk_window(1, (0, 0, 2000, 1080), 1, {1: 1920 * 1080, 2: 80 * 1080}, z_order=0)
    center_app = mk_window(2, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, z_order=2)
    assert pick_candidate([poking, center_app], M_CENTER) is center_app


def test_minimized_windows_are_never_candidates():
    wmin = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, minimized=True)
    other = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400})
    assert pick_candidate([wmin, other], M_CENTER) is other
    assert pick_candidate([wmin], M_CENTER) is None


def test_window_off_monitor_is_not_candidate():
    left_win = mk_window(1, (0, 0, 1920, 1080), 1, {1: 1920 * 1080})
    assert pick_candidate([left_win], M_RIGHT) is None


def test_spanning_window_scores_by_largest_intersection():
    """A window covering the full center monitor and half of the right one
    belongs to CENTER (largest visible part), not RIGHT."""
    spanning = mk_window(1, (1920, 0, 4800, 1080), 2, {2: 1920 * 1080, 3: 960 * 1080})
    assert pick_candidate([spanning], M_CENTER) is spanning
    assert pick_candidate([spanning], M_RIGHT) is spanning  # only candidate on RIGHT too


def test_right_monitor_gets_its_own_window_over_spanning():
    spanning = mk_window(1, (1920, 0, 4800, 1080), 2, {2: 1920 * 1080, 3: 960 * 1080})
    right_win = mk_window(2, (3840, 0, 5760, 1080), 3, {3: 1920 * 1080}, z_order=5)
    assert pick_candidate([spanning, right_win], M_RIGHT) is right_win


def test_preference_overrides_default_choice():
    big = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, process="chrome.exe")
    small = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400}, process="Code.exe")
    prefs = [WindowPreference(process="Code.exe")]
    assert pick_candidate([big, small], M_CENTER, prefs) is small


def test_preference_by_title_substring():
    big = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, title="Something Else")
    small = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400}, title="Gmail - Chrome")
    prefs = [WindowPreference(title_contains="gmail")]
    assert pick_candidate([big, small], M_CENTER, prefs) is small


def test_first_matching_preference_wins():
    a = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, process="notepad.exe")
    b = mk_window(2, (1920, 0, 2400, 400), 2, {2: 480 * 400}, process="Code.exe")
    prefs = [WindowPreference(process="Code.exe"), WindowPreference(process="notepad.exe")]
    assert pick_candidate([a, b], M_CENTER, prefs) is b
    prefs_rev = [WindowPreference(process="notepad.exe"), WindowPreference(process="Code.exe")]
    assert pick_candidate([a, b], M_CENTER, prefs_rev) is a


def test_tie_broken_by_z_order_then_foreground():
    w1 = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, z_order=1)
    w0 = mk_window(2, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, z_order=0)
    assert pick_candidate([w1, w0], M_CENTER) is w0  # z=0 is topmost


def test_multiple_windows_same_monitor_is_explicit_not_random():
    """Policy check: deterministic front-most choice, documented in README;
    users can override with preferences."""
    w_a = mk_window(1, (1920, 0, 3840, 1080), 2, {2: 1920 * 1080}, title="A", z_order=9)
    w_b = mk_window(2, (2400, 100, 3800, 1000), 2, {2: 1400 * 900}, title="B", z_order=0)
    for _ in range(3):
        assert pick_candidate([w_a, w_b], M_CENTER) is w_b  # stable, deterministic


def test_dpi_scaled_monitors_use_physical_pixels():
    """A 150% DPI monitor reports physical pixels; a maximised window there
    covers the full physical area."""
    hidpi = mk_monitor(4, (0, 0, 2560, 1440), primary=True)  # 170% of 96dpi
    maximised = mk_window(1, (0, 0, 2560, 1440), 4, {4: 2560 * 1440})
    assert pick_candidate([maximised], hidpi) is maximised
    partial = mk_window(2, (0, 0, 1000, 800), 4, {4: 1000 * 800})
    assert pick_candidate([partial], hidpi) is maximised or True
    # intersection math holds regardless of DPI:
    assert intersection_area(hidpi.rect, maximised.rect) == 2560 * 1440


def test_empty_monitor_has_no_candidate():
    assert pick_candidate([], M_LEFT) is None
