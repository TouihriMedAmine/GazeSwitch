"""Temporal selection state machine: raw classifications -> stable monitor.

Implements the stability requirements for monitor selection:

* **Temporal smoothing** - a sliding vote window over the last N frames.
* **Confidence** - a candidate must hold >= ``min_confidence`` of the votes.
* **Dwell time** - a *new* candidate must persist >= ``dwell_ms`` before the
  selection actually switches.
* **Hysteresis** - the dwell + vote requirements must be met again to switch
  away; the classifier's ambiguous band provides spatial hysteresis.
* **UNKNOWN** - forced when tracking is lost for >= ``lost_ms``; returning
  from UNKNOWN requires the normal vote+dwell procedure, so a single noisy
  frame can never change the selection.
"""
from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from .monitor_classifier import Classification, MonitorState

log = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    window_frames: int = 7
    min_confidence: float = 0.6  # fraction of the window needed for a candidate
    dwell_ms: int = 350
    lost_ms: int = 600
    # > 0: vote over the frames of the last N ms instead of the last
    # window_frames frames, so latency does not depend on the camera FPS.
    vote_window_ms: int = 0


@dataclass
class SelectionEvent:
    state: MonitorState
    previous: MonitorState
    changed: bool
    stable: bool  # state != UNKNOWN and has been selected for >= dwell_ms
    votes: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0


class SelectionStateMachine:
    def __init__(self, config: Optional[SelectionConfig] = None) -> None:
        self.cfg = config or SelectionConfig()
        maxlen = None if self.cfg.vote_window_ms > 0 else max(1, self.cfg.window_frames)
        self.window: Deque[MonitorState] = deque(maxlen=maxlen)
        self._window_times: Deque[int] = deque(maxlen=maxlen)
        self.selected: MonitorState = MonitorState.UNKNOWN
        self._pending: Optional[MonitorState] = None
        self._pending_since_ms: int = 0
        self._switched_at_ms: int = 0
        self._lost_since_ms: Optional[int] = None
        self._last_confidence = 0.0

    # ------------------------------------------------------------------ API
    def update(
        self, classification: Classification, tracking_ok: bool, now_ms: int
    ) -> SelectionEvent:
        previous = self.selected
        self._last_confidence = classification.confidence

        if not tracking_ok:
            if self._lost_since_ms is None:
                self._lost_since_ms = now_ms
            if (
                self.selected is not MonitorState.UNKNOWN
                and now_ms - self._lost_since_ms >= self.cfg.lost_ms
            ):
                self._force_unknown(now_ms)
        else:
            self._lost_since_ms = None
            self._feed(classification.state, now_ms)

        votes = dict(Counter(self.window))
        stable = (
            self.selected is not MonitorState.UNKNOWN
            and now_ms - self._switched_at_ms >= self.cfg.dwell_ms
        )
        return SelectionEvent(
            state=self.selected,
            previous=previous,
            changed=self.selected is not previous,
            stable=stable,
            votes={k.value: v for k, v in votes.items()},
            confidence=self._last_confidence,
        )

    def reset(self) -> None:
        self.window.clear()
        self._window_times.clear()
        self._pending = None
        self._lost_since_ms = None
        self._switched_at_ms = 0

    @property
    def time_in_state_ms(self) -> int:
        return self._switched_at_ms

    # ------------------------------------------------------------- internals
    def _feed(self, state: MonitorState, now_ms: int) -> None:
        self.window.append(state)
        self._window_times.append(now_ms)
        if self.cfg.vote_window_ms > 0:
            while self._window_times and now_ms - self._window_times[0] > self.cfg.vote_window_ms:
                self._window_times.popleft()
                self.window.popleft()
        candidate = self._majority()
        if candidate is None or candidate is self.selected:
            self._pending = None
            return
        # A new candidate state must persist before switching.
        if self._pending is not candidate:
            self._pending = candidate
            self._pending_since_ms = now_ms
            return
        if now_ms - self._pending_since_ms >= self.cfg.dwell_ms:
            log.info("Selection: %s -> %s", self.selected.value, candidate.value)
            self.selected = candidate
            self._switched_at_ms = now_ms
            self._pending = None

    def _majority(self) -> Optional[MonitorState]:
        if not self.window:
            return None
        state, count = Counter(self.window).most_common(1)[0]
        if count / len(self.window) >= self.cfg.min_confidence:
            return state
        return None

    def _force_unknown(self, now_ms: int) -> None:
        if self.selected is MonitorState.UNKNOWN:
            return
        log.info("Selection: %s -> UNKNOWN (tracking lost)", self.selected.value)
        self.selected = MonitorState.UNKNOWN
        self._switched_at_ms = now_ms
        self._pending = None
        self.window.clear()
        self._window_times.clear()
