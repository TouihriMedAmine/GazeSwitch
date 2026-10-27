"""Targeting controller safety tests: arming gate, emergency stop, typing
guard, candidate stability, backoff. Focus requests are mocked (no real
Win32 focus changes in tests)."""
from __future__ import annotations

import pytest

from src.monitor_classifier import MonitorState
from src.selection_state import SelectionEvent
from src.target_manager import MonitorTargetRegistry
from src.targeting import TargetingConfig, TargetingController


class FakeWindow:
    def __init__(self, hwnd: int):
        self.hwnd = hwnd


class FakeRegistry:
    """Minimal stand-in for MonitorTargetRegistry (candidate_for returns
    objects exposing .hwnd, like the real WindowInfo)."""

    def __init__(self, hwnd_for_state=None):
        self.hwnd_for_state = {
            state: (FakeWindow(hwnd) if hwnd is not None else None)
            for state, hwnd in (hwnd_for_state or {}).items()
        }

    def candidate_for(self, state):
        return self.hwnd_for_state.get(state)


def ev(state, changed=False):
    return SelectionEvent(state=state, previous=state, changed=changed, stable=True,
                          votes={}, confidence=1.0)


def make_controller(gate=True, input_age_ms=10_000, focus_calls=None):
    focus_calls = focus_calls if focus_calls is not None else []
    controller = TargetingController(
        TargetingConfig(typing_quiet_ms=700, target_dwell_ms=800, focus_backoff_s=0.0),
        enabled_gate=gate,
        input_age_fn=lambda: input_age_ms,
    )

    def fake_set_foreground(hwnd):
        focus_calls.append(hwnd)
        return True  # pretend Windows granted foreground

    controller._set_foreground = fake_set_foreground
    return controller, focus_calls


def test_toggle_refused_without_gate():
    controller, _ = make_controller(gate=False)
    assert "turned off in settings" in controller.toggle()
    assert not controller.armed


def test_arm_disarm_cycle():
    controller, _ = make_controller(gate=True)
    assert controller.toggle() == "Switching on. Ctrl+Alt+K stops it."
    assert controller.armed
    assert controller.toggle() == "Switching off"
    assert not controller.armed


def test_emergency_stop_always_disarms():
    controller, _ = make_controller(gate=True)
    controller.toggle()
    message = controller.emergency_off()
    assert not controller.armed
    assert message.startswith("Stopped")


def test_update_without_arm_does_nothing():
    controller, calls = make_controller(gate=True)  # never armed
    registry = FakeRegistry({MonitorState.CENTER: 42})
    result = controller.update(ev(MonitorState.CENTER, changed=True), registry, 1000)
    assert result is None
    assert calls == []


def test_candidate_must_be_stable_before_focus():
    controller, calls = make_controller()
    controller.toggle()
    registry = FakeRegistry({MonitorState.CENTER: 42})
    # t=1000: candidate first seen -> not stable yet
    r1 = controller.update(ev(MonitorState.CENTER), registry, 1000)
    # t=1500: 500ms < 800ms dwell -> still unstable
    r2 = controller.update(ev(MonitorState.CENTER), registry, 1500)
    # t=2000: 1000ms >= 800ms -> focus now
    r3 = controller.update(ev(MonitorState.CENTER), registry, 2000)
    from src.targeting import FocusResult

    assert r1 is FocusResult.SKIPPED_CANDIDATE_UNSTABLE
    assert r2 is FocusResult.SKIPPED_CANDIDATE_UNSTABLE
    assert r3 is FocusResult.FOCUSED
    assert calls == [42]
    # no repeated trigger while the same candidate stays
    assert controller.update(ev(MonitorState.CENTER), registry, 2500) is None


def test_candidate_change_resets_stability():
    controller, calls = make_controller()
    controller.toggle()
    registry = FakeRegistry({MonitorState.CENTER: 42})
    controller.update(ev(MonitorState.CENTER), registry, 0)
    controller.update(ev(MonitorState.CENTER), registry, 500)
    registry.hwnd_for_state[MonitorState.CENTER] = FakeWindow(99)  # candidate changed
    from src.targeting import FocusResult

    result = controller.update(ev(MonitorState.CENTER), registry, 600)
    assert result is FocusResult.SKIPPED_CANDIDATE_UNSTABLE
    result = controller.update(ev(MonitorState.CENTER), registry, 1500)  # 900ms >= dwell
    assert result is FocusResult.FOCUSED
    assert calls == [99]


def test_unknown_state_clears_tracking():
    controller, calls = make_controller()
    controller.toggle()
    registry = FakeRegistry({MonitorState.CENTER: 42})
    controller.update(ev(MonitorState.CENTER), registry, 0)
    from src.targeting import FocusResult

    result = controller.update(ev(MonitorState.UNKNOWN), registry, 100)
    assert result is None
    # back to CENTER: stability restarts
    result = controller.update(ev(MonitorState.CENTER), registry, 200)
    assert result is FocusResult.SKIPPED_CANDIDATE_UNSTABLE


def test_no_candidate_is_skipped():
    controller, calls = make_controller()
    controller.toggle()
    registry = FakeRegistry({})  # nothing on this monitor
    from src.targeting import FocusResult

    result = controller.update(ev(MonitorState.RIGHT), registry, 1000)
    assert result is FocusResult.SKIPPED_NO_CANDIDATE
    assert calls == []


def test_typing_guard_blocks_focus(monkeypatch):
    controller, calls = make_controller(input_age_ms=200)  # typed 200ms ago
    controller.toggle()
    registry = FakeRegistry({MonitorState.CENTER: 42})
    controller.update(ev(MonitorState.CENTER), registry, 0)
    controller.update(ev(MonitorState.CENTER), registry, 500)
    from src.targeting import FocusResult

    result = controller.update(ev(MonitorState.CENTER), registry, 1000)
    assert result is FocusResult.SKIPPED_TYPING
    assert calls == []


def test_gate_blocks_even_when_config_true_but_cli_missing():
    controller = TargetingController(enabled_gate=False)
    assert not controller.armed
    assert "turned off in settings" in controller.toggle()
    assert not controller.armed
