"""Monitor -> candidate application mapping.

Given enumerated windows and the monitor->slot mapping, choose the *candidate*
application for each monitor slot.

Policy (MVP, explicit rather than silent guessing):
* ``preferences`` (optional, per-slot, from config.yaml): a window matching a
  preference (process name exact-match and/or title substring) always wins.
* Otherwise the **front-most window** on that monitor wins (Z-order: the
  one you used last there); ties are broken by visible area, then by
  foreground state.

Only windows that live mainly on the monitor count (a window spanning
several monitors belongs to the one showing most of it), and slivers under
``min_area_fraction`` of the monitor are ignored, so a small always-on-top
widget doesn't steal the selection. Both filters fall back when they would
leave nothing. Minimised / hidden / cloaked windows are never candidates
(they are filtered at enumeration).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .monitor_classifier import MonitorState
from .window_manager import MonitorInfo, WindowInfo

log = logging.getLogger(__name__)


@dataclass
class WindowPreference:
    process: Optional[str] = None      # exact base filename, e.g. "Code.exe"
    title_contains: Optional[str] = None

    def matches(self, window: WindowInfo) -> bool:
        if self.process is not None:
            if window.process_name.lower() != self.process.lower():
                return False
        if self.title_contains is not None:
            if self.title_contains.lower() not in window.title.lower():
                return False
        return self.process is not None or self.title_contains is not None


def pick_candidate(
    windows: List[WindowInfo],
    monitor: MonitorInfo,
    preferences: Optional[List[WindowPreference]] = None,
    min_area_fraction: float = 0.03,
) -> Optional[WindowInfo]:
    """Choose the candidate window for one monitor. Pure function (tested)."""
    prefs = preferences or []
    on_monitor = [
        w
        for w in windows
        if w.visible and not w.minimized and w.areas.get(monitor.handle, 0) > 0
    ]
    if not on_monitor:
        return None
    x0, y0, x1, y1 = monitor.rect
    min_area = min_area_fraction * max(1, (x1 - x0) * (y1 - y0))
    big_enough = [w for w in on_monitor if w.areas.get(monitor.handle, 0) >= min_area]
    at_home = [w for w in big_enough
               if w.areas.get(monitor.handle, 0) >= max(w.areas.values())]
    candidates = at_home or big_enough or on_monitor

    def pref_rank(w: WindowInfo) -> int:
        for i, p in enumerate(prefs):
            if p.matches(w):
                return i
        return len(prefs)

    candidates.sort(
        key=lambda w: (
            pref_rank(w),
            w.z_order,
            -w.areas.get(monitor.handle, 0),
            0 if w.foreground else 1,
        )
    )
    return candidates[0]


class MonitorTargetRegistry:
    """Throttled re-enumeration plus per-slot candidate tracking."""

    def __init__(
        self,
        window_manager,  # WindowManager; untyped to keep module import-safe
        refresh_s: float = 2.0,
        preferences: Optional[Dict[str, List[WindowPreference]]] = None,
    ) -> None:
        self.wm = window_manager
        self.refresh_s = refresh_s
        self.preferences = preferences or {}
        self.slots: Dict[str, MonitorInfo] = {}
        self.windows: List[WindowInfo] = []
        self.candidates: Dict[str, Optional[WindowInfo]] = {
            s.value: None for s in MonitorState if s is not MonitorState.UNKNOWN
        }
        self._last_refresh_ms: int = -10**9
        self._on_target_change: Dict[str, Callable[[Optional[WindowInfo]], None]] = {}

    # ------------------------------------------------------------------ API
    def register_on_target_change(
        self, slot: MonitorState, callback: Callable[[Optional[WindowInfo]], None]
    ) -> None:
        self._on_target_change[slot.value] = callback

    def update(
        self,
        now_ms: int,
        slots: Dict[str, MonitorInfo],
        force: bool = False,
    ) -> bool:
        """Re-enumerate windows at most every ``refresh_s``. Returns True if
        the candidate set changed this call."""
        if not force and (now_ms - self._last_refresh_ms) < self.refresh_s * 1000:
            self.slots = slots
            return False
        self._last_refresh_ms = now_ms
        self.slots = slots

        try:
            monitors = list(self.slots.values())
            self.windows = self.wm.enumerate_windows(monitors)
        except Exception as exc:  # noqa: BLE001 - enumeration must never kill the loop
            log.error("Window enumeration failed: %s", exc)
            return False

        changed = False
        for slot_name in (MonitorState.LEFT, MonitorState.CENTER, MonitorState.RIGHT):
            monitor = self.slots.get(slot_name.value)
            candidate = (
                pick_candidate(self.windows, monitor, self.preferences.get(slot_name.value))
                if monitor is not None
                else None
            )
            previous = self.candidates.get(slot_name.value)
            self.candidates[slot_name.value] = candidate
            if (previous is None) != (candidate is None) or (
                previous is not None
                and candidate is not None
                and (previous.hwnd != candidate.hwnd or previous.title != candidate.title)
            ):
                changed = True
                cb = self._on_target_change.get(slot_name.value)
                if cb is not None:
                    try:
                        cb(candidate)
                    except Exception as exc:  # noqa: BLE001
                        log.error("target-change callback failed: %s", exc)
        return changed

    def candidate_for(self, state: MonitorState) -> Optional[WindowInfo]:
        if state is MonitorState.UNKNOWN:
            return None
        return self.candidates.get(state.value)

    def description(self, state: MonitorState) -> str:
        w = self.candidate_for(state)
        if w is None:
            return "none"
        return f"{w.process_name or 'unknown'} - {w.title}"
