"""System-tray icon and menu (pystray).

The icon runs its own message loop in a daemon thread. Menu callbacks never
touch the tracking pipeline directly: they queue a request name that the
main loop consumes once per frame (like the global hotkeys), so all side
effects stay on the main thread. The main loop pushes a small state snapshot
back with :meth:`TrayController.update_state`, which refreshes the icon
colour, tooltip and menu checkmarks only when something changed.

The tray shows assets/icon.png. State (switching on, watching only, needs
calibration) is in the tooltip and the menu, not in the icon colour.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from . import app_env

log = logging.getLogger(__name__)

COLOR_ARMED = (46, 204, 113)
COLOR_OBSERVING = (52, 152, 219)
COLOR_UNCALIBRATED = (243, 156, 18)


_ICON_MASTER = None


def make_icon_image(color=None, size: int = 64):  # noqa: ARG001 - kept for callers
    """App icon from assets/icon.png, resized for the tray or the exe."""
    global _ICON_MASTER
    from PIL import Image

    if _ICON_MASTER is None:
        _ICON_MASTER = Image.open(app_env.resolve_resource("assets/icon.png")).convert("RGBA")
    if _ICON_MASTER.size == (size, size):
        return _ICON_MASTER.copy()
    return _ICON_MASTER.resize((size, size), Image.LANCZOS)


@dataclass(frozen=True)
class TrayState:
    monitor: str = "UNKNOWN"
    armed: bool = False
    gate: bool = False
    calibrated: bool = False
    preview: bool = False
    calibrating: bool = False
    cameras: Tuple[str, ...] = ()
    camera_choice: str = ""  # "" = automatic
    camera_label: str = ""  # camera actually in use
    silent: bool = False


class TrayController:
    def __init__(self, shortcuts: Optional[Dict[str, str]] = None) -> None:
        self.shortcuts = shortcuts or {}  # request name -> "Ctrl+Alt+P"
        self._requests: "queue.Queue[str]" = queue.Queue()
        self._state = TrayState()
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="tray", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def poll(self) -> Optional[str]:
        try:
            return self._requests.get_nowait()
        except queue.Empty:
            return None

    def notify(self, message: str, title: str = app_env.APP_NAME) -> None:
        if self._icon is None:
            return
        try:
            self._icon.notify(message, title)
        except Exception as exc:  # noqa: BLE001
            log.debug("Tray notification failed: %s", exc)

    def update_state(self, **changes) -> None:
        new = replace(self._state, **changes)
        if new == self._state:
            return
        old, self._state = self._state, new
        if self._icon is None:
            return
        if self._color(old) != self._color(new):
            self._icon.icon = make_icon_image(self._color(new))
        self._icon.title = self._tooltip(new)
        self._icon.update_menu()

    # ------------------------------------------------------------- internals
    @staticmethod
    def _color(state: TrayState):
        if not state.calibrated:
            return COLOR_UNCALIBRATED
        return COLOR_ARMED if state.armed else COLOR_OBSERVING

    @staticmethod
    def _tooltip(state: TrayState) -> str:
        if state.calibrating:
            return f"{app_env.APP_NAME} - calibrating..."
        if not state.calibrated:
            return f"{app_env.APP_NAME} - needs calibration"
        mode = "switching ON" if state.armed else "watching only"
        return f"{app_env.APP_NAME} - {mode} - {state.monitor}"

    def _request(self, name: str):
        def handler(icon, item) -> None:  # noqa: ARG001 - pystray signature
            self._requests.put(name)
        return handler

    def _toggle_autostart(self, icon, item) -> None:  # noqa: ARG002
        try:
            app_env.set_autostart(not app_env.autostart_enabled())
        except OSError as exc:
            log.error("Could not change start-with-Windows setting: %s", exc)
        icon.update_menu()

    def _camera_items(self):
        """Camera submenu, rebuilt from the latest state on every menu update."""
        import pystray

        item = pystray.MenuItem
        state = self._state
        auto = "Automatic"
        if not state.camera_choice and state.camera_label:
            auto = f"Automatic ({state.camera_label})"
        items = [item(auto, self._request("camera:"),
                      checked=lambda _: not self._state.camera_choice, radio=True)]
        for name in state.cameras:
            items.append(item(
                name, self._request(f"camera:{name}"),
                checked=lambda _, n=name: self._state.camera_choice == n, radio=True,
            ))
        if not state.cameras:
            items.append(item("No cameras found", None, enabled=False))
        items += [pystray.Menu.SEPARATOR,
                  item(self._label("Next camera", "next_camera"), self._request("next_camera")),
                  item("Refresh list", self._request("refresh_cameras"))]
        return items

    @staticmethod
    def _open(path) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            log.error("Could not open %s: %s", path, exc)

    def _label(self, text: str, request: str) -> str:
        """Menu text with its global shortcut right-aligned (Win32 tab stop)."""
        shortcut = self.shortcuts.get(request)
        return f"{text}\t{shortcut}" if shortcut else text

    def _build_menu(self):
        import pystray

        item = pystray.MenuItem
        return pystray.Menu(
            item(lambda _: f"Looking at: {self._state.monitor}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            item(
                self._label("Switch windows by gaze", "toggle_targeting"),
                self._request("toggle_targeting"),
                checked=lambda _: self._state.armed,
                enabled=lambda _: self._state.gate,
                default=True,
            ),
            item(self._label("Calibrate", "calibrate"), self._request("calibrate"),
                 enabled=lambda _: not self._state.calibrating),
            item(self._label("Show camera preview", "toggle_preview"),
                 self._request("toggle_preview"), checked=lambda _: self._state.preview),
            item(self._label("Silent mode", "toggle_silent"), self._request("toggle_silent"),
                 checked=lambda _: self._state.silent),
            item("Camera", pystray.Menu(self._camera_items)),
            pystray.Menu.SEPARATOR,
            item("Start with Windows", self._toggle_autostart,
                 checked=lambda _: app_env.autostart_enabled()),
            item("Open settings file",
                 lambda icon, it: self._open(app_env.app_dir() / "config.yaml")),
            item("Open logs folder",
                 lambda icon, it: self._open(app_env.app_dir() / "logs")),
            pystray.Menu.SEPARATOR,
            item(self._label("Quit GazeSwitch", "quit"), self._request("quit")),
        )

    def _run(self) -> None:
        try:
            import pystray

            self._icon = pystray.Icon(
                app_env.APP_NAME,
                make_icon_image(self._color(self._state)),
                self._tooltip(self._state),
                menu=self._build_menu(),
            )
        except Exception as exc:  # noqa: BLE001 - tray is optional; app keeps running
            log.error("Could not create tray icon: %s", exc)
            self._ready.set()
            return
        self._ready.set()
        self._icon.run()
