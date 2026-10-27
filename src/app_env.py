"""Desktop-application environment: portable paths, single instance,
start-with-Windows registration and native message boxes.

The packaged app is a portable folder::

    GazeSwitch/
        GazeSwitch.exe
        config.yaml        (created on first run from the bundled default)
        baselines/, logs/  (created on demand)
        _internal/         (PyInstaller runtime + bundled model)

All relative paths in config.yaml resolve against the folder holding the
exe (or the project root when running from source).
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

APP_NAME = "GazeSwitch"
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_ERROR_ALREADY_EXISTS = 183


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Writable folder for config, baselines and logs."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """Read-only folder holding bundled files (model, default config)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return app_dir()


def resolve_resource(path: str) -> str:
    """Return ``path`` if it exists (relative to the app folder), otherwise
    the bundled copy when there is one."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    bundled = resource_dir() / p
    return str(bundled) if bundled.exists() else str(p)


def ensure_user_config(name: str = "config.yaml") -> Path:
    """Copy the bundled default config next to the exe on first run."""
    target = app_dir() / name
    if not target.exists():
        default = resource_dir() / name
        if default.exists() and default != target:
            shutil.copyfile(default, target)
            log.info("Created %s from the bundled default", target)
    return target


# ---------------------------------------------------------------- instance
_mutex_handle: Optional[int] = None


def acquire_single_instance() -> bool:
    """True if no other GazeSwitch is running (keeps a named mutex open)."""
    global _mutex_handle
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, f"Local\\{APP_NAME}SingleInstance")
    if not handle:
        return True
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return False
    _mutex_handle = handle
    return True


def message_box(text: str, title: str = APP_NAME, error: bool = False) -> None:
    if sys.platform != "win32":
        print(text, file=sys.stderr)
        return
    import ctypes

    flags = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(None, text, title, flags)


# --------------------------------------------------------------- autostart
def autostart_command() -> str:
    if is_frozen():
        return f'"{Path(sys.executable).resolve()}"'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    interpreter = pythonw if pythonw.exists() else Path(sys.executable)
    return f'"{interpreter}" "{app_dir() / "main.py"}" --tray'


def autostart_enabled() -> bool:
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            return bool(value)
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        if enabled:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, autostart_command())
            log.info("Start with Windows enabled")
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
            log.info("Start with Windows disabled")
