"""Action system: extensible callbacks fired on *stable* monitor transitions.

Guarantees:
* ``on_enter`` / ``on_exit`` fire exactly once per actual transition - never
  repeatedly while the selection stays on the same monitor (deduplicated
  here, independently of the state machine).
* UNKNOWN only triggers ``on_exit`` of a concrete monitor; monitor actions
  never fire for UNKNOWN entries.
* The default registered action only logs - no input is injected anywhere.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .monitor_classifier import CONCRETE_STATES, MonitorState
from .selection_state import SelectionEvent

log = logging.getLogger(__name__)


class MonitorAction(ABC):
    """Base class for monitor-transition actions (future extension point)."""

    @abstractmethod
    def on_enter(self, state: MonitorState, target_description: str) -> None:
        ...

    @abstractmethod
    def on_exit(self, state: MonitorState) -> None:
        ...


class TransitionLogAction(MonitorAction):
    """Default action: structured logging of monitor transitions."""

    def on_enter(self, state: MonitorState, target_description: str) -> None:
        log.info("Monitor entered: %s (candidate target: %s)", state.value, target_description)

    def on_exit(self, state: MonitorState) -> None:
        log.info("Monitor exited: %s", state.value)


class ActionRouter:
    def __init__(self) -> None:
        self.actions: List[MonitorAction] = []
        self._current: Optional[MonitorState] = None
        self._last_target: str = ""

    def register(self, action: MonitorAction) -> None:
        self.actions.append(action)

    def update(
        self,
        event: SelectionEvent,
        target_description: Callable[[], str],
    ) -> bool:
        """Process a selection event. Fires actions only on real transitions.

        Returns True if a transition was dispatched.
        """
        if event.state is self._current:
            return False  # dedupe: no repeated triggers for the same monitor

        previous = self._current
        self._current = event.state

        if previous in CONCRETE_STATES:
            for action in self.actions:
                try:
                    action.on_exit(previous)
                except Exception as exc:  # noqa: BLE001
                    log.error("on_exit(%s) failed: %s", previous.value, exc)

        if event.state in CONCRETE_STATES:
            desc = target_description()
            self._last_target = desc
            for action in self.actions:
                try:
                    action.on_enter(event.state, desc)
                except Exception as exc:  # noqa: BLE001
                    log.error("on_enter(%s) failed: %s", event.state.value, exc)
            return True
        return False

    @property
    def current(self) -> Optional[MonitorState]:
        return self._current
