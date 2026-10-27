"""ActionRouter tests: single-fire transitions, dedupe, UNKNOWN handling,
and the default observe-only (log-only) behaviour."""
from __future__ import annotations

from src.action_router import ActionRouter, MonitorAction, TransitionLogAction
from src.monitor_classifier import MonitorState
from src.selection_state import SelectionEvent


def event(state: MonitorState, changed: bool, stable: bool = True) -> SelectionEvent:
    return SelectionEvent(
        state=state, previous=state if not changed else MonitorState.UNKNOWN,
        changed=changed, stable=stable, votes={}, confidence=0.9,
    )


class RecordingAction(MonitorAction):
    def __init__(self) -> None:
        self.entered: list = []
        self.exited: list = []

    def on_enter(self, state, target_description):
        self.entered.append((state, target_description))

    def on_exit(self, state):
        self.exited.append(state)


def target_desc() -> str:
    return "Code.exe - main.py"


def test_enter_fires_once_per_transition():
    router = ActionRouter()
    rec = RecordingAction()
    router.register(rec)

    router.update(event(MonitorState.CENTER, changed=True), target_desc)
    router.update(event(MonitorState.CENTER, changed=False), target_desc)
    router.update(event(MonitorState.CENTER, changed=False), target_desc)

    assert rec.entered == [(MonitorState.CENTER, "Code.exe - main.py")]
    assert rec.exited == []


def test_no_repeated_triggers_while_monitor_unchanged():
    router = ActionRouter()
    rec = RecordingAction()
    router.register(rec)
    for _ in range(50):
        router.update(event(MonitorState.LEFT, changed=False), target_desc)
    assert len(rec.entered) == 1


def test_transition_fires_exit_then_enter():
    router = ActionRouter()
    rec = RecordingAction()
    router.register(rec)
    router.update(event(MonitorState.LEFT, changed=True), target_desc)
    router.update(event(MonitorState.RIGHT, changed=True), target_desc)
    assert rec.exited == [MonitorState.LEFT]
    assert rec.entered == [
        (MonitorState.LEFT, "Code.exe - main.py"),
        (MonitorState.RIGHT, "Code.exe - main.py"),
    ]


def test_unknown_only_fires_exit():
    router = ActionRouter()
    rec = RecordingAction()
    router.register(rec)
    router.update(event(MonitorState.CENTER, changed=True), target_desc)
    router.update(event(MonitorState.UNKNOWN, changed=True), target_desc)
    assert rec.exited == [MonitorState.CENTER]
    assert [s for s, _ in rec.entered] == [MonitorState.CENTER]

    # staying UNKNOWN: nothing further fires
    router.update(event(MonitorState.UNKNOWN, changed=False), target_desc)
    assert len(rec.entered) == 1


def test_reenter_after_unknown_fires_again():
    router = ActionRouter()
    rec = RecordingAction()
    router.register(rec)
    router.update(event(MonitorState.CENTER, changed=True), target_desc)
    router.update(event(MonitorState.UNKNOWN, changed=True), target_desc)
    router.update(event(MonitorState.CENTER, changed=True), target_desc)
    assert [s for s, _ in rec.entered] == [MonitorState.CENTER, MonitorState.CENTER]
    # single exit when leaving CENTER for UNKNOWN; UNKNOWN->CENTER fires no exit
    assert rec.exited == [MonitorState.CENTER]


def test_action_exception_does_not_break_router():
    class Broken(MonitorAction):
        def on_enter(self, state, target_description):
            raise RuntimeError("boom")

        def on_exit(self, state):
            raise RuntimeError("boom")

    router = ActionRouter()
    rec = RecordingAction()
    router.register(Broken())
    router.register(rec)
    router.update(event(MonitorState.CENTER, changed=True), target_desc)  # must not raise
    assert len(rec.entered) == 1


def test_default_router_has_log_only_action():
    """Default registration in the app is TransitionLogAction (observe-only);
    the router itself ships with no input-injecting action."""
    router = ActionRouter()
    router.register(TransitionLogAction())
    assert len(router.actions) == 1
    assert isinstance(router.actions[0], TransitionLogAction)
