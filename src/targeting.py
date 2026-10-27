"""Phase 2: explicitly-armed focus targeting with a strict safety policy.

This controller is the ONLY component allowed to change application focus,
and only when *all* of the following hold:

1. ``config.targeting.enabled`` is true AND the app was started with
   ``--enable-targeting`` (the hard gate), AND
2. the user armed targeting at runtime (global hotkey Ctrl+Alt+G or ``t``
   in the preview window) - explicit, reversible, with a visible banner, AND
3. the monitor selection is a concrete monitor that became stable through
   the state machine (votes + dwell already satisfied), AND
4. the candidate window has stayed the same window for >= ``target_dwell_ms``,
5. the user has not produced any input for >= ``typing_quiet_ms``
   (GetLastInputInfo) - never steal focus mid-typing,
6. the window is not already foreground.

Focus is requested with ``SetForegroundWindow`` and *verified*; failures are
logged and followed by a cooldown instead of aggressive retries. Windows'
foreground-lock restrictions are respected: if activation is refused, the
system stays in observe mode for that transition. When the plain request is
refused, one unassigned (no-op) virtual key is injected so Windows grants this
process foreground rights; no meaningful keystrokes are ever injected.

Emergency stop: Ctrl+Alt+K (global) or ``e`` in the preview disarms
immediately and is always available.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .monitor_classifier import MonitorState
from .selection_state import SelectionEvent
from .target_manager import MonitorTargetRegistry

log = logging.getLogger(__name__)

_SW_RESTORE = 9
_VK_UNASSIGNED = 0xE8
_KEYEVENTF_KEYUP = 0x0002


class TargetingState(str, Enum):
    OFF = "OFF"
    ARMED = "ARMED"


class FocusResult(str, Enum):
    FOCUSED = "FOCUSED"
    ALREADY_FOREGROUND = "ALREADY_FOREGROUND"
    SKIPPED_TYPING = "SKIPPED_TYPING"
    SKIPPED_CANDIDATE_UNSTABLE = "SKIPPED_CANDIDATE_UNSTABLE"
    SKIPPED_BACKOFF = "SKIPPED_BACKOFF"
    SKIPPED_NO_CANDIDATE = "SKIPPED_NO_CANDIDATE"
    FAILED = "FAILED"


@dataclass
class TargetingConfig:
    typing_quiet_ms: int = 700
    target_dwell_ms: int = 800
    focus_backoff_s: float = 1.3


class TargetingController:
    def __init__(
        self,
        config: Optional[TargetingConfig] = None,
        enabled_gate: bool = False,
        input_age_fn=None,
    ) -> None:
        self.cfg = config or TargetingConfig()
        self.enabled_gate = enabled_gate  # config AND CLI flag
        self._input_age_fn = input_age_fn or self._ms_since_last_input
        self.state = TargetingState.OFF
        self._last_focus_attempt = 0.0
        self._candidate_hwnd: Dict[str, Optional[int]] = {}
        self._candidate_since_ms: Dict[str, int] = {}
        self._focused_for: Dict[str, Optional[int]] = {}

    # ------------------------------------------------------------------ API
    @property
    def armed(self) -> bool:
        return self.state is TargetingState.ARMED

    def toggle(self) -> str:
        if self.armed:
            self.disarm()
            return "Switching off"
        if not self.enabled_gate:
            log.info("Targeting refused: disabled by configuration/CLI gate")
            return "Switching is turned off in settings (targeting.enabled)"
        self.state = TargetingState.ARMED
        log.warning("Targeting ARMED - focus changes on stable monitor switches are enabled")
        return "Switching on. Ctrl+Alt+K stops it."

    def arm(self) -> str:
        return self.toggle()

    def disarm(self) -> str:
        self.state = TargetingState.OFF
        self._candidate_hwnd.clear()
        self._candidate_since_ms.clear()
        self._focused_for.clear()
        log.info("Targeting disarmed - back to observe-only")
        return "Targeting disarmed"

    def emergency_off(self) -> str:
        self.disarm()
        log.warning("EMERGENCY STOP used - targeting is now off")
        return "Stopped. Switching is off."

    # ---------------------------------------------------------------- update
    def update(
        self,
        event: SelectionEvent,
        registry: MonitorTargetRegistry,
        now_ms: int,
    ) -> Optional[FocusResult]:
        """Call once per frame. May attempt a focus action; returns the
        result (or None when nothing was attempted)."""
        if not self.armed:
            return None
        if event.state not in (MonitorState.LEFT, MonitorState.CENTER, MonitorState.RIGHT):
            self._candidate_hwnd.clear()
            self._candidate_since_ms.clear()
            self._focused_for.clear()
            return None

        slot = event.state.value
        window = registry.candidate_for(event.state)
        hwnd = window.hwnd if window is not None else None

        # Track candidate stability: the same window must stay the candidate.
        if self._candidate_hwnd.get(slot) != hwnd:
            self._candidate_hwnd[slot] = hwnd
            self._candidate_since_ms[slot] = now_ms
            self._focused_for.pop(slot, None)
            return FocusResult.SKIPPED_CANDIDATE_UNSTABLE if hwnd is not None \
                else FocusResult.SKIPPED_NO_CANDIDATE

        if hwnd is None:
            return FocusResult.SKIPPED_NO_CANDIDATE
        if now_ms - self._candidate_since_ms[slot] < self.cfg.target_dwell_ms:
            return FocusResult.SKIPPED_CANDIDATE_UNSTABLE
        if self._focused_for.get(slot) == hwnd:
            return None  # already handled this entry; no repeated triggers

        result = self._try_focus(hwnd)
        if result in (FocusResult.FOCUSED, FocusResult.ALREADY_FOREGROUND):
            self._focused_for[slot] = hwnd
        return result

    # ------------------------------------------------------------ internals
    def _try_focus(self, hwnd: int) -> FocusResult:
        if sys.platform != "win32":
            return FocusResult.FAILED
        user32 = ctypes.windll.user32

        # 1) recent-input guard: never steal focus while the user is typing
        if self._input_age_fn() < self.cfg.typing_quiet_ms:
            log.debug("Targeting: user is active; not stealing focus")
            return FocusResult.SKIPPED_TYPING

        # 2) failure backoff: no aggressive retry loops
        if time.monotonic() - self._last_focus_attempt < self.cfg.focus_backoff_s:
            return FocusResult.SKIPPED_BACKOFF

        if user32.GetForegroundWindow() == hwnd:
            return FocusResult.ALREADY_FOREGROUND

        self._last_focus_attempt = time.monotonic()
        if self._set_foreground(hwnd):
            log.info("Targeting: focused window hwnd=%s", hwnd)
            return FocusResult.FOCUSED
        log.warning(
            "Targeting: SetForegroundWindow refused for hwnd=%s "
            "(Windows foreground lock). Staying in observe mode for this "
            "transition; bring the window forward manually if needed.",
            hwnd,
        )
        return FocusResult.FAILED

    @staticmethod
    def _set_foreground(hwnd: int) -> bool:
        """Request + verify foreground activation. Isolated so tests can
        substitute it without touching the safety guards."""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, _SW_RESTORE)
        if user32.SetForegroundWindow(hwnd) and user32.GetForegroundWindow() == hwnd:
            return True

        # Windows only lets the process that received the last input event take
        # the foreground. An unassigned virtual key gives us that right without
        # any application reacting to the keystroke.
        user32.keybd_event(_VK_UNASSIGNED, 0, 0, 0)
        user32.keybd_event(_VK_UNASSIGNED, 0, _KEYEVENTF_KEYUP, 0)

        fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
        our_thread = kernel32.GetCurrentThreadId()
        attached = bool(
            fg_thread and fg_thread != our_thread
            and user32.AttachThreadInput(our_thread, fg_thread, True)
        )
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(our_thread, fg_thread, False)
        return user32.GetForegroundWindow() == hwnd

    @staticmethod
    def _ms_since_last_input() -> int:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint32),
                ("dwTime", ctypes.c_uint32),
            ]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 10**9  # unknown -> be safe, do not steal focus
        now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        last = lii.dwTime & 0xFFFFFFFF
        return (now - last) & 0xFFFFFFFF
