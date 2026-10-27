"""Win32 window and monitor enumeration via ctypes (no pywin32 dependency).

Responsibilities:
* per-monitor-DPI awareness (physical-pixel rectangles everywhere),
* enumerate visible top-level windows with title/class/process/rect,
* enumerate monitors and map them to LEFT / CENTER / RIGHT slots by their
  geometric x-order (no resolution or arrangement is hardcoded),
* per-window monitor assignment by largest visible intersection area
  (handles windows spanning multiple monitors).

Everything ctypes-related is bound lazily at call time so importing this
module is safe on any OS (tests only use the pure dataclasses/helpers).
"""
from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
MONITOR_DEFAULTTONEAREST = 2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MDT_EFFECTIVE_DPI = 0


class WindowsOnlyError(RuntimeError):
    pass


def _user32():
    if sys.platform != "win32":
        raise WindowsOnlyError("Window enumeration requires Windows")
    return ctypes.windll.user32


def _dwmapi():
    if sys.platform != "win32":
        raise WindowsOnlyError("Window enumeration requires Windows")
    return ctypes.windll.dwmapi


def _shcore():
    if sys.platform != "win32":
        raise WindowsOnlyError("Window enumeration requires Windows")
    return ctypes.windll.shcore


def _kernel32():
    if sys.platform != "win32":
        raise WindowsOnlyError("Window enumeration requires Windows")
    return ctypes.windll.kernel32


# --------------------------------------------------------------------- pure
@dataclass
class MonitorInfo:
    handle: int
    device: str
    rect: Tuple[int, int, int, int]  # physical pixels (left, top, right, bottom)
    work_rect: Tuple[int, int, int, int]
    primary: bool
    dpi: int

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    process_name: str
    pid: int
    rect: Tuple[int, int, int, int]
    monitor_handle: int
    areas: Dict[int, int] = field(default_factory=dict)  # monitor handle -> px area
    minimized: bool = False
    visible: bool = True
    foreground: bool = False
    z_order: int = 0  # lower = closer to the top of the Z order

    @property
    def area(self) -> int:
        return max(0, self.rect[2] - self.rect[0]) * max(0, self.rect[3] - self.rect[1])


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def map_monitors_to_slots(
    monitors: List[MonitorInfo],
) -> Tuple[Dict[str, MonitorInfo], str]:
    """Assign monitors to LEFT / CENTER / RIGHT slots by x-position.

    Works exactly for the 3-monitor case. Other counts get a documented
    best-effort mapping and a warning string.
    """
    from .monitor_classifier import MonitorState  # local: avoid import cycle noise

    slots: Dict[str, MonitorInfo] = {}
    if not monitors:
        return slots, "no monitors detected"
    ordered = sorted(monitors, key=lambda m: (m.rect[0], m.rect[1]))
    primary = next((m for m in monitors if m.primary), ordered[0])

    if len(ordered) == 3:
        slots[MonitorState.LEFT.value] = ordered[0]
        slots[MonitorState.CENTER.value] = ordered[1]
        slots[MonitorState.RIGHT.value] = ordered[2]
        note = "3 monitors mapped by x-order"
    elif len(ordered) == 2:
        slots[MonitorState.LEFT.value] = ordered[0]
        slots[MonitorState.RIGHT.value] = ordered[1]
        slots[MonitorState.CENTER.value] = primary
        note = "2 monitors: CENTER maps to the primary monitor (gaze CENTER is ambiguous)"
    elif len(ordered) == 1:
        for key in ("LEFT", "CENTER", "RIGHT"):
            slots[key] = ordered[0]
        note = "single monitor: all states map to it"
    else:
        slots[MonitorState.LEFT.value] = ordered[0]
        slots[MonitorState.RIGHT.value] = ordered[-1]
        slots[MonitorState.CENTER.value] = primary
        note = f"{len(ordered)} monitors: using left-most, right-most and primary"
    return slots, note


# ------------------------------------------------------------------- ctypes
class WindowManager:
    """Enumerates monitors and visible top-level windows."""

    def __init__(self, excluded_processes: Optional[List[str]] = None) -> None:
        self.excluded_processes = {
            e.lower() for e in (excluded_processes or [])
        }
        self._own_pid = 0
        if sys.platform == "win32":
            self._own_pid = int(_kernel32().GetCurrentProcessId())

    # ---------------------------------------------------------------- DPI
    @staticmethod
    def set_process_dpi_awareness() -> str:
        """Make GetWindowRect / monitor rects physical pixels (per-monitor).

        Must be called before any window is created. Returns a description
        of what succeeded; failures are non-fatal (coordinates then follow
        the process' virtualisation, which is logged).
        """
        if sys.platform != "win32":
            return "non-windows platform"
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (void*)-4
            if _user32().SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return "per-monitor-v2"
        except Exception:  # noqa: BLE001 - older Windows
            pass
        try:
            if _shcore().SetProcessDpiAwareness(2):  # PER_MONITOR_DPI_AWARE
                return "per-monitor"
        except Exception:  # noqa: BLE001
            pass
        try:
            _user32().SetProcessDPIAware()
            return "system-aware"
        except Exception:  # noqa: BLE001
            return "unaware (failed to set; coordinates may be virtualised)"

    # ----------------------------------------------------------- monitors
    def enumerate_monitors(self) -> List[MonitorInfo]:
        user32 = _user32()
        monitors: List[MonitorInfo] = []

        # Win32 callback: BOOL MonitorEnumProc(HMONITOR, HDC, LPRECT, LPARAM)
        proto = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,  # HMONITOR
            ctypes.c_void_p,  # HDC
            ctypes.c_void_p,  # LPRECT
            ctypes.c_void_p,  # LPARAM
        )

        def callback(hmon: int, _dc: int, _rect: int, _data: int) -> int:
            info = self._monitor_info(hmon)
            if info is not None:
                monitors.append(info)
            return 1

        cb = proto(callback)
        user32.EnumDisplayMonitors(None, None, cb, 0)
        return monitors

    def _monitor_info(self, hmon: int) -> Optional[MonitorInfo]:
        user32 = _user32()

        class RECT_(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint32),
                ("rcMonitor", RECT_),
                ("rcWork", RECT_),
                ("dwFlags", ctypes.c_uint32),
                ("szDevice", ctypes.c_wchar * 32),
            ]

        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return None
        dpi = 96
        try:
            dx = ctypes.c_uint32(0)
            dy = ctypes.c_uint32(0)
            if _shcore().GetDpiForMonitor(
                ctypes.c_void_p(hmon), MDT_EFFECTIVE_DPI, ctypes.byref(dx), ctypes.byref(dy)
            ) == 0:
                dpi = int(dx.value)
        except Exception:  # noqa: BLE001 - Win 8.1- fallback
            pass
        rect = (mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom)
        work = (mi.rcWork.left, mi.rcWork.top, mi.rcWork.right, mi.rcWork.bottom)
        return MonitorInfo(
            handle=int(hmon),
            device=mi.szDevice,
            rect=rect,
            work_rect=work,
            primary=bool(mi.dwFlags & 1),
            dpi=dpi,
        )

    # ----------------------------------------------------------- windows
    def enumerate_windows(self, monitors: List[MonitorInfo]) -> List[WindowInfo]:
        """Visible top-level windows with per-monitor intersection areas.

        EnumWindows yields windows in Z-order (top first), which we keep as
        ``z_order`` for candidate tie-breaking.
        """
        user32 = _user32()
        windows: List[WindowInfo] = []
        order = [0]

        proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd: int, _lparam: int) -> int:
            try:
                info = self._window_info(hwnd, monitors)
            except Exception as exc:  # noqa: BLE001 - never break enumeration
                log.debug("skipping hwnd %s: %s", hwnd, exc)
                return 1
            if info is not None:
                info.z_order = order[0]
                order[0] += 1
                windows.append(info)
            return 1

        cb = proto(callback)
        user32.EnumWindows(cb, 0)

        fg = self.foreground_hwnd()
        for w in windows:
            w.foreground = w.hwnd == fg
        return windows

    def _window_info(
        self, hwnd: int, monitors: List[MonitorInfo]
    ) -> Optional[WindowInfo]:
        user32 = _user32()
        if not user32.IsWindowVisible(hwnd):
            return None

        exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if exstyle & WS_EX_TOOLWINDOW:
            return None

        # Skip UWP suspended/"cloaked" windows (invisible despite IsWindowVisible).
        cloaked = ctypes.c_uint32(0)
        if _dwmapi().DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        ) == 0 and cloaked.value != 0:
            return None

        title = self._window_text(hwnd)
        if not title:
            return None

        rect = self._window_rect(hwnd)
        if rect is None or rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
            return None

        pid = ctypes.c_uint32(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        pid_val = int(pid.value)
        if pid_val == self._own_pid:
            return None  # never list our own preview window

        process_name = self._process_name(pid_val)
        if process_name.lower() in self.excluded_processes:
            return None

        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)

        areas = {m.handle: intersection_area(rect, m.rect) for m in monitors}
        monitor_handle = user32.MonitorFromWindow(
            hwnd, MONITOR_DEFAULTTONEAREST
        )
        return WindowInfo(
            hwnd=int(hwnd),
            title=title,
            class_name=class_buf.value,
            process_name=process_name,
            pid=pid_val,
            rect=rect,
            monitor_handle=int(monitor_handle),
            areas=areas,
            minimized=bool(user32.IsIconic(hwnd)),
            visible=True,
        )

    @staticmethod
    def _window_text(hwnd: int) -> str:
        user32 = _user32()
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.strip()

    @staticmethod
    def _window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        class RECT_(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT_()
        if not _user32().GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)

    @staticmethod
    def _process_name(pid: int) -> str:
        try:
            kernel32 = _kernel32()
            hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not hproc:
                return ""
            try:
                size = ctypes.c_uint32(1024)
                buf = ctypes.create_unicode_buffer(1024)
                if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
                    full = buf.value
                    return full.rsplit("\\", 1)[-1]
                return ""
            finally:
                kernel32.CloseHandle(hproc)
        except Exception:  # noqa: BLE001
            return ""

    def foreground_hwnd(self) -> Optional[int]:
        if sys.platform != "win32":
            return None
        hwnd = _user32().GetForegroundWindow()
        return int(hwnd) if hwnd else None
