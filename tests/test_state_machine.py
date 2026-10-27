"""State machine tests: smoothing, dwell time, hysteresis, tracking loss,
UNKNOWN handling, and no single-frame switching."""
from __future__ import annotations

import pytest

from src.monitor_classifier import Classification, MonitorState
from src.selection_state import SelectionConfig, SelectionStateMachine


def cls(state: MonitorState, conf: float = 0.8) -> Classification:
    return Classification(state=state, confidence=conf, x=0.0)


CFG = SelectionConfig(window_frames=5, min_confidence=0.6, dwell_ms=300, lost_ms=500)


def test_first_stable_state_is_adopted_after_dwell():
    sm = SelectionStateMachine(CFG)
    t = 0
    ev = sm.update(cls(MonitorState.CENTER), True, t)
    assert ev.state is MonitorState.UNKNOWN  # not yet selected
    for i in range(1, 10):
        t = i * 100
        ev = sm.update(cls(MonitorState.CENTER), True, t)
    assert ev.state is MonitorState.CENTER
    # switch happened after dwell_ms of stable votes
    assert ev.changed is False  # already switched before this event
    assert ev.stable is True


def test_single_noisy_frame_never_changes_selection():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.CENTER), True, i * 100)
    assert sm.selected is MonitorState.CENTER
    # one isolated LEFT frame (1 of 5 votes = 0.2 < 0.6 confidence)
    ev = sm.update(cls(MonitorState.LEFT), True, 1000)
    assert ev.state is MonitorState.CENTER
    assert ev.changed is False


def test_switch_requires_dwell_time():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.CENTER), True, i * 100)
    assert sm.selected is MonitorState.CENTER

    # LEFT votes start at t=1000; the sliding window still holds a CENTER
    # majority until the 3rd LEFT frame, then dwell_ms must elapse.
    sm.update(cls(MonitorState.LEFT), True, 1100)
    assert sm.selected is MonitorState.CENTER
    sm.update(cls(MonitorState.LEFT), True, 1200)
    assert sm.selected is MonitorState.CENTER
    ev = sm.update(cls(MonitorState.LEFT), True, 1301)  # LEFT becomes candidate
    assert sm.selected is MonitorState.CENTER
    sm.update(cls(MonitorState.LEFT), True, 1400)
    sm.update(cls(MonitorState.LEFT), True, 1500)
    ev = sm.update(cls(MonitorState.LEFT), True, 1601)  # dwell satisfied
    assert sm.selected is MonitorState.LEFT
    assert ev.changed is True
    assert ev.previous is MonitorState.CENTER


def test_oscillation_never_switches():
    """Alternating LEFT/RIGHT never reaches the vote threshold."""
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 40):
        state = MonitorState.LEFT if i % 2 else MonitorState.RIGHT
        ev = sm.update(cls(state), True, i * 50)
        assert ev.state is not MonitorState.LEFT
        assert ev.state is not MonitorState.RIGHT


def test_tracking_loss_forces_unknown_after_lost_ms():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.CENTER), True, i * 100)
    assert sm.selected is MonitorState.CENTER

    ev = sm.update(cls(MonitorState.UNKNOWN), False, 1100)
    assert ev.state is MonitorState.CENTER  # under lost_ms -> keep last state
    ev = sm.update(cls(MonitorState.UNKNOWN), False, 1601)
    assert ev.state is MonitorState.UNKNOWN
    assert ev.changed is True


def test_short_tracking_loss_does_not_clear_state():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.CENTER), True, i * 100)
    sm.update(cls(MonitorState.UNKNOWN), False, 1200)  # brief loss
    ev = sm.update(cls(MonitorState.CENTER), True, 1300)
    assert ev.state is MonitorState.CENTER


def test_brief_blip_does_not_switch():
    """3 CENTER frames inside a LEFT stream (3/5 = 0.6 >= threshold? no:
    window is the last 5 votes -> blip must not reach majority+dwell)."""
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.LEFT), True, i * 100)
    assert sm.selected is MonitorState.LEFT
    # brief CENTER blip: 2 frames only
    sm.update(cls(MonitorState.CENTER), True, 1000)
    sm.update(cls(MonitorState.CENTER), True, 1100)
    ev = sm.update(cls(MonitorState.LEFT), True, 1200)
    assert ev.state is MonitorState.LEFT


def test_recovery_from_unknown_requires_dwell():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 10):
        sm.update(cls(MonitorState.CENTER), True, i * 100)
    sm.update(cls(MonitorState.UNKNOWN), False, 2000)
    sm.update(cls(MonitorState.UNKNOWN), False, 3000)
    assert sm.selected is MonitorState.UNKNOWN
    # single LEFT frame right after recovery
    ev = sm.update(cls(MonitorState.LEFT), True, 3100)
    assert ev.state is MonitorState.UNKNOWN


def test_votes_are_exposed():
    sm = SelectionStateMachine(CFG)
    t = 0
    for i in range(1, 6):
        ev = sm.update(cls(MonitorState.CENTER), True, i * 100)
    assert ev.votes == {"CENTER": 5}


def test_min_confidence_blocks_split_windows():
    """3 LEFT + 2 CENTER votes = 0.6 -> LEFT candidate wins at exactly the
    threshold; 0.59 would not."""
    cfg = SelectionConfig(window_frames=5, min_confidence=0.6, dwell_ms=100, lost_ms=500)
    sm = SelectionStateMachine(cfg)
    sm.update(cls(MonitorState.LEFT), True, 0)
    sm.update(cls(MonitorState.LEFT), True, 10)
    sm.update(cls(MonitorState.LEFT), True, 20)
    sm.update(cls(MonitorState.CENTER), True, 30)
    ev = sm.update(cls(MonitorState.CENTER), True, 200)
    assert ev.state is MonitorState.LEFT  # 3/5 == 0.6 -> adopted
