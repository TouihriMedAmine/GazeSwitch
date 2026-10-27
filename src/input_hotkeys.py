"""Global hotkeys via Win32 RegisterHotKey (no extra dependencies).

Runs a message loop in a daemon thread. Registration failures (e.g. the
combo is taken by another app) are non-fatal: the preview-window keys
remain available as fallback. The thread only *sets events*; the main loop
polls them once per frame, so all side effects stay on the main thread.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
}


def parse_combo(combo: str) -> Optional[Tuple[int, int]]:
    """Parse 'ctrl+alt+g' into (modifiers, virtual-key code)."""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        return None
    mods = 0
    vk: Optional[int] = None
    for part in parts:
        if part in _MODIFIERS:
            mods |= _MODIFIERS[part]
        elif len(part) == 1 and "a" <= part <= "z":
            vk = ord(part.upper())
        elif len(part) == 1 and "0" <= part <= "9":
            vk = ord(part)
        elif len(part) >= 2 and part[0] == "f" and part[1:].isdigit():
            n = int(part[1:])
            if 1 <= n <= 24:
                vk = 0x70 + n - 1
        else:
            log.warning("Unsupported key '%s' in hotkey combo '%s'", part, combo)
            return None
    if vk is None:
        return None
    return mods, vk


def format_combo(combo: str) -> str:
    """'ctrl+alt+p' -> 'Ctrl+Alt+P' (for menus and messages)."""
    names = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    return "+".join(names.get(p, p.upper()) for p in parts)


class HotkeyMonitor:
    """Registers global hotkeys and exposes thread-safe consume() events."""

    def __init__(self, combos: Dict[str, str]) -> None:
        self.combos = combos  # name -> combo string
        self._events: Dict[str, threading.Event] = {
            name: threading.Event() for name in combos
        }
        self._thread: Optional[threading.Thread] = None
        self._thread_id: int = 0
        self.registered: Dict[str, bool] = {}

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()

    def consume(self, name: str) -> bool:
        event = self._events.get(name)
        if event is not None and event.is_set():
            event.clear()
            return True
        return False

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        self._thread = None

    # ------------------------------------------------------------- internals
    def _run(self) -> None:
        import ctypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        ids: Dict[int, str] = {}
        for hotkey_id, (name, combo) in enumerate(self.combos.items(), start=1):
            parsed = parse_combo(combo)
            if parsed is None:
                self.registered[name] = False
                log.warning("Hotkey '%s' disabled: cannot parse combo '%s'", name, combo)
                continue
            mods, vk = parsed
            if user32.RegisterHotKey(None, hotkey_id, mods, vk):
                ids[hotkey_id] = name
                self.registered[name] = True
                log.info("Global hotkey registered: %s (%s)", combo, name)
            else:
                self.registered[name] = False
                log.warning(
                    "Could not register hotkey %s (id=%d) - it may be in use "
                    "by another application. Preview-window keys still work.",
                    combo,
                    hotkey_id,
                )

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", ctypes.c_void_p),
                ("message", ctypes.c_uint32),
                ("wParam", ctypes.c_void_p),
                ("lParam", ctypes.c_void_p),
                ("time", ctypes.c_uint32),
                ("pt_x", ctypes.c_long),
                ("pt_y", ctypes.c_long),
            ]

        msg = MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                name = ids.get(int(msg.wParam or 0))
                if name is not None:
                    self._events[name].set()
        for hotkey_id in ids:
            user32.UnregisterHotKey(None, hotkey_id)
