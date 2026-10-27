"""Camera preview UI.

The camera image is the content layer and stays full-bleed. Controls float
above it as glass panels (blurred video behind a dark tint with a light
edge), following Apple's Liquid Glass rules translated to OpenCV/Pillow:

* glass only for the floating controls, never stacked on other glass;
  elements sitting on glass (keycaps, monitor glyphs) use plain fills,
* one accent colour, reserved for the selected monitor,
* hierarchy from type weight and label colours (primary / secondary /
  tertiary) instead of borders and boxes,
* diagnostics behind progressive disclosure (Details, key D).

Text is Segoe UI Variable - Windows' system face, which like San Francisco
has Text/Display optical sizes - falling back to static Segoe UI.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .monitor_classifier import MonitorState

# Dark-appearance system colours (RGB / RGBA)
LABEL = (255, 255, 255, 255)
SECONDARY = (235, 235, 245, 165)
TERTIARY = (235, 235, 245, 100)
BLUE = (10, 132, 255)
GREEN = (48, 209, 88)
ORANGE = (255, 159, 10)
GRAY = (152, 152, 157)
GLASS_TINT = (24, 24, 28, 120)
GLASS_EDGE = (255, 255, 255, 50)
ON_GLASS_FILL = (255, 255, 255, 34)

_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
_SLOTS = (MonitorState.LEFT, MonitorState.CENTER, MonitorState.RIGHT)
_BASE_WIDTH = 720  # layout metrics below are designed at this canvas width


@dataclass
class PreviewView:
    """Everything the preview shows for one frame."""

    state: MonitorState = MonitorState.UNKNOWN
    stable: bool = False
    calibrated: bool = False
    camera_ok: bool = True
    face: bool = False
    armed: bool = False
    target: str = ""
    x: float = 0.0
    center_threshold: float = 0.12
    side_threshold: float = 0.32
    landmarks: Optional[np.ndarray] = None
    message: Optional[str] = None
    calibrating: bool = False
    calib_progress: float = 0.0
    calib_valid: int = 0
    calib_needed: int = 0
    calib_step: int = 0
    calib_steps: int = 1
    calib_target: str = ""
    calib_collecting: bool = False
    calib_targets: Tuple[str, ...] = ()
    calib_coverage: Tuple[bool, ...] = ()  # head directions reached, from the right, clockwise
    calib_countdown: float = 0.0  # seconds until collection starts
    calib_flash: bool = False  # previous monitor just finished
    details: bool = False
    detail_lines: List[str] = field(default_factory=list)
    close_label: str = "Quit"
    silent: bool = False
    camera_name: str = ""  # camera in use
    cameras: Tuple[str, ...] = ()
    camera_choice: str = ""  # "" = automatic
    camera_menu: bool = False
    pointer: Optional[Tuple[int, int]] = None  # mouse position, for hover


def window_label(description: str) -> str:
    """'Cursor.exe - main.py - Cursor' -> 'main.py - Cursor'."""
    if not description or description.strip().lower() in ("none", "n/a"):
        return ""
    process, _, title = description.partition(" - ")
    if title.strip():
        return title.strip()
    return process.removesuffix(".exe").strip()


def headline(view: PreviewView) -> Tuple[str, str]:
    """Title + subtitle for the bottom card."""
    if not view.camera_ok:
        return "Waiting for the camera", "Make sure the camera (or Camo) is connected"
    if not view.face:
        return "Can't see your face", "Face the camera in good light"
    if not view.calibrated:
        return "Calibrate to start", "Look at your center monitor, then press C"
    if view.state is MonitorState.UNKNOWN:
        return "Looking away", "Look at a monitor to select it"
    label = window_label(view.target)
    return f"{view.state.value.title()} monitor", label or "No window on this monitor"


_WINDOW_ICONS: List[int] = []


def _hicon(image) -> int:
    """Win32 icon handle from an RGBA image, so a window can show it."""
    import ctypes
    from ctypes import wintypes

    image = image.convert("RGBA")
    width, height = image.size

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class ICONINFO(ctypes.Structure):
        _fields_ = [
            ("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD),
            ("yHotspot", wintypes.DWORD), ("hbmMask", wintypes.HBITMAP),
            ("hbmColor", wintypes.HBITMAP),
        ]

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC, ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
    ]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    gdi32.CreateBitmap.restype = wintypes.HBITMAP
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    user32.CreateIconIndirect.argtypes = [ctypes.POINTER(ICONINFO)]
    user32.CreateIconIndirect.restype = wintypes.HICON

    header = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0)
    bits = ctypes.c_void_p()
    hdc = user32.GetDC(None)
    color = gdi32.CreateDIBSection(hdc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
    user32.ReleaseDC(None, hdc)
    if not bits:
        return 0
    ctypes.memmove(bits, image.tobytes("raw", "BGRA"), width * height * 4)
    mask = gdi32.CreateBitmap(width, height, 1, 1, None)
    icon = user32.CreateIconIndirect(ctypes.byref(ICONINFO(True, 0, 0, mask, color)))
    gdi32.DeleteObject(color)
    gdi32.DeleteObject(mask)
    return int(ctypes.cast(icon, ctypes.c_void_p).value or 0)


def _window_icons() -> List[int]:
    if not _WINDOW_ICONS:
        from .tray import make_icon_image

        _WINDOW_ICONS.extend(_hicon(make_icon_image(size=size)) for size in (16, 32))
    return _WINDOW_ICONS


def style_window(title: str, topmost: bool = False) -> None:
    """Dark title bar (Windows 10 2004+ / 11), the app icon in the top-left,
    and optional always-on-top."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return
    value = ctypes.c_int(1)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(
        hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)  # DWMWA_USE_IMMERSIVE_DARK_MODE
    )
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = wintypes.LPARAM
    for kind, handle in enumerate(_window_icons()):  # 0 = title bar, 1 = Alt-Tab
        if handle:
            user32.SendMessageW(hwnd, 0x0080, kind, handle)  # WM_SETICON
    if topmost:
        HWND_TOPMOST = ctypes.c_void_p(-1)
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
        user32.SetWindowPos(ctypes.c_void_p(hwnd), HWND_TOPMOST, 0, 0, 0, 0,
                            SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)


class PreviewRenderer:
    def __init__(self, width: int = _BASE_WIDTH) -> None:
        self.width = max(320, int(width))
        self._fonts: Dict[Tuple[int, int, bool], ImageFont.FreeTypeFont] = {}
        self._masks: Dict[tuple, Image.Image] = {}
        self._fills: Dict[tuple, Image.Image] = {}
        self._crop: Optional[Tuple[float, float, float]] = None
        self._hits: List[Tuple[str, Tuple[int, int, int, int]]] = []

    def hit_test(self, x: int, y: int) -> Optional[str]:
        """Name of the clickable element under (x, y) in the last frame."""
        for name, (x0, y0, x1, y1) in reversed(self._hits):
            if x0 <= x < x1 and y0 <= y < y1:
                return name
        return None

    # ---------------------------------------------------------------- render
    def render(self, frame_bgr: np.ndarray, view: PreviewView) -> np.ndarray:
        self._hits = []
        if view.calibrating:
            return self._render_calibration(frame_bgr, view)
        self._crop = None
        h0, w0 = frame_bgr.shape[:2]
        W = self.width
        H = max(1, int(round(h0 * W / w0)))
        self._s = W / _BASE_WIDTH
        interp = cv2.INTER_AREA if w0 > W else cv2.INTER_LINEAR
        frame = cv2.resize(frame_bgr, (W, H), interpolation=interp)
        if view.calibrating or not view.face:
            frame = cv2.convertScaleAbs(frame, alpha=0.55)

        small = cv2.resize(frame, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        blurred = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

        self._base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._blur = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        self._layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._layer)
        self._W, self._H = W, H

        if view.face and view.landmarks is not None and not view.calibrating:
            # Face markers belong to the content layer: flatten them into the
            # video so the glass panels drawn next cover them.
            self._face(view)
            self._base = Image.alpha_composite(self._base.convert("RGBA"), self._layer).convert("RGB")
            self._layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            self._draw = ImageDraw.Draw(self._layer)
        top = self._top_bar(view)
        if view.message:
            top = self._toast(view.message, top)
        if view.details and view.detail_lines:
            self._details(view.detail_lines, top)
        self._bottom_card(view)
        if view.camera_menu:
            # The menu floats above everything: flatten first so it isn't
            # glass stacked on glass.
            self._base = Image.alpha_composite(self._base.convert("RGBA"), self._layer).convert("RGB")
            self._layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            self._draw = ImageDraw.Draw(self._layer)
            self._camera_menu(view)

        out = Image.alpha_composite(self._base.convert("RGBA"), self._layer).convert("RGB")
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------- layout
    def _top_bar(self, v: PreviewView) -> int:
        S = self._px
        m, hcap = S(14), S(30)

        if not v.calibrated:
            text, dot = "Not calibrated", ORANGE
        elif v.armed:
            text, dot = "Switching on", GREEN
        else:
            text, dot = "Watching only", GRAY
        if v.silent:
            text += " · Silent"
        f = self._font(S(13), 600)
        tw = self._draw.textlength(text, font=f)
        w = int(S(12) + S(8) + S(8) + tw + S(14))
        self._glass((m, m, m + w, m + hcap), hcap // 2)
        cy = m + hcap // 2
        r = S(4)
        self._circle(m + S(12) + r, cy, r, dot + (255,))
        self._draw.text((m + S(12) + S(8) + S(8), cy), text, font=f, fill=LABEL, anchor="lm")
        left_end = self._camera_pill(v, m + w + S(8), m, hcap)

        keys = [("C", "Calibrate"), ("D", "Details"), ("Q", v.close_label)]
        fk, fl = self._font(S(11), 600), self._font(S(12), 400)
        cap = S(18)
        widths = [cap + S(6) + self._draw.textlength(label, font=fl) for _, label in keys]
        total = int(S(12) * 2 + sum(widths) + S(14) * (len(keys) - 1))
        x0 = self._W - m - total
        if x0 > left_end + S(8):  # skip the hints when the window is too narrow
            self._glass((x0, m, self._W - m, m + hcap), hcap // 2)
            x = x0 + S(12)
            for (key, label), kw in zip(keys, widths):
                self._fill((x, cy - cap // 2, x + cap, cy + cap - cap // 2), S(5), ON_GLASS_FILL)
                self._draw.text((x + cap / 2, cy), key, font=fk, fill=LABEL, anchor="mm")
                self._draw.text((x + cap + S(6), cy), label, font=fl, fill=SECONDARY, anchor="lm")
                x += kw + S(14)
        return m + hcap

    def _hovered(self, v: PreviewView, box) -> bool:
        if v.pointer is None:
            return False
        x, y = v.pointer
        return box[0] <= x < box[2] and box[1] <= y < box[3]

    def _camera_pill(self, v: PreviewView, x0: int, top: int, hcap: int) -> int:
        """Capsule with the camera in use; click opens the camera menu."""
        S = self._px
        f = self._font(S(13), 400)
        name = self._ellipsize(v.camera_name or "No camera", f, S(150))
        tw = self._draw.textlength(name, font=f)
        w = int(S(12) + S(16) + S(8) + tw + S(10) + S(8) + S(12))
        box = (x0, top, x0 + w, top + hcap)
        self._glass(box, hcap // 2)
        if v.camera_menu or self._hovered(v, box):
            self._fill(box, hcap // 2, (255, 255, 255, 30))
        cy = top + hcap / 2
        gx = x0 + S(12)
        self._fill((gx, cy - S(5), gx + S(11), cy + S(5)), S(2), SECONDARY)  # body
        self._fill((gx + S(12), cy - S(3), gx + S(16), cy + S(3)), S(1), SECONDARY)  # lens
        tx = gx + S(16) + S(8)
        self._draw.text((tx, cy), name, font=f, fill=LABEL, anchor="lm")
        chx = tx + tw + S(10) + S(4)
        d = S(3)
        pts = ([(chx - d, cy + d / 2), (chx, cy - d / 2), (chx + d, cy + d / 2)] if v.camera_menu
               else [(chx - d, cy - d / 2), (chx, cy + d / 2), (chx + d, cy - d / 2)])
        self._polyline(pts, SECONDARY, max(1, S(1.6)))
        self._hits.append(("camera_menu", box))
        return x0 + w

    def _camera_menu(self, v: PreviewView) -> None:
        """Drop-down list of cameras under the camera pill."""
        pill = next((b for n, b in self._hits if n == "camera_menu"), None)
        if pill is None:
            return
        S = self._px
        f = self._font(S(13), 400)
        auto = "Automatic"
        if not v.camera_choice and v.camera_name:
            auto = f"Automatic ({v.camera_name})"
        items = [("camera:", auto)] + [(f"camera:{n}", n) for n in v.cameras]
        items.append(("refresh_cameras", "Refresh list"))
        pad, row, indent = S(6), S(30), S(30)
        max_w = self._W - pill[0] - S(14)
        w = int(min(max_w, max(self._draw.textlength(t, font=f) for _, t in items)
                    + indent + S(16) + 2 * pad))
        x0, y0 = pill[0], pill[3] + S(6)
        sep = S(9)
        h = 2 * pad + row * len(items) + sep
        self._glass((x0, y0, x0 + w, y0 + h), S(14))
        self._hits.append(("menu_body", (x0, y0, x0 + w, y0 + h)))  # under the rows
        y = y0 + pad
        for i, (name, label) in enumerate(items):
            if name == "refresh_cameras":
                self._fill((x0 + pad + S(8), y + sep // 2, x0 + w - pad - S(8), y + sep // 2 + 1),
                           0, (255, 255, 255, 40))
                y += sep
            rbox = (x0 + pad, y, x0 + w - pad, y + row)
            hovered = self._hovered(v, rbox)
            if hovered:
                self._fill(rbox, S(8), BLUE + (255,))
            selected = name.startswith("camera:") and name[len("camera:"):] == v.camera_choice
            cy = y + row / 2
            if selected:
                cx = rbox[0] + S(14)
                self._polyline([(cx - S(5), cy), (cx - S(1.5), cy + S(4)), (cx + S(5), cy - S(4))],
                               LABEL, max(1, S(1.8)))
            color = LABEL if hovered or name != "refresh_cameras" else SECONDARY
            text = self._ellipsize(label, f, rbox[2] - rbox[0] - indent - S(8))
            self._draw.text((rbox[0] + indent, cy), text, font=f, fill=color, anchor="lm")
            self._hits.append((name, rbox))
            y += row

    def _toast(self, message: str, top: int) -> int:
        S = self._px
        f = self._font(S(13), 400)
        hcap = S(32)
        text = self._ellipsize(message, f, self._W - S(28) - S(32))
        w = int(self._draw.textlength(text, font=f) + S(32))
        x0 = (self._W - w) // 2
        y0 = top + S(10)
        self._glass((x0, y0, x0 + w, y0 + hcap), hcap // 2)
        self._draw.text((self._W / 2, y0 + hcap / 2), text, font=f, fill=LABEL, anchor="mm")
        return y0 + hcap

    def _details(self, lines: List[str], top: int) -> None:
        S = self._px
        f = self._font(S(12), 400, mono=True)
        pad, lh = S(12), S(18)
        w = int(max(self._draw.textlength(t, font=f) for t in lines) + 2 * pad)
        x0, y0 = S(14), top + S(10)
        self._glass((x0, y0, x0 + w, y0 + 2 * pad + lh * len(lines) - S(4)), S(14))
        for i, line in enumerate(lines):
            self._draw.text((x0 + pad, y0 + pad + i * lh), line, font=f, fill=SECONDARY)

    def _bottom_card(self, v: PreviewView) -> None:
        S = self._px
        m, ch, pad = S(14), S(80), S(18)
        x0, y0, x1, y1 = m, self._H - m - ch, self._W - m, self._H - m
        self._glass((x0, y0, x1, y1), S(22))

        gw, gh, gap = S(50), S(30), S(12)
        mw = 3 * gw + 2 * gap
        gx0 = x1 - pad - mw
        gy = y0 + S(14)

        title, subtitle = headline(v)
        text_w = gx0 - S(16) - (x0 + pad)
        ft, fs = self._font(S(20), 600), self._font(S(13), 400)
        cy = (y0 + y1) / 2
        self._draw.text((x0 + pad, cy - S(1)), self._ellipsize(title, ft, text_w),
                        font=ft, fill=LABEL, anchor="ls")
        self._draw.text((x0 + pad, cy + S(7)), self._ellipsize(subtitle, fs, text_w),
                        font=fs, fill=SECONDARY, anchor="lt")

        fl = self._font(S(11), 600)
        active = v.face and v.calibrated and v.camera_ok
        centers = []
        for i, slot in enumerate(_SLOTS):
            gx = gx0 + i * (gw + gap)
            box = (gx, gy, gx + gw, gy + gh)
            centers.append(gx + gw / 2)
            selected = active and v.state is slot
            if selected and v.stable:
                self._fill(box, S(6), BLUE + (255,))
                stand, label = BLUE + (255,), LABEL
            elif selected:
                self._fill(box, S(6), BLUE + (70,))
                self._ring(box, S(6), BLUE + (255,), S(2))
                stand, label = BLUE + (200,), LABEL
            else:
                self._fill(box, S(6), ON_GLASS_FILL)
                self._ring(box, S(6), (255, 255, 255, 60), 1)
                stand, label = (255, 255, 255, 60), TERTIARY if not active else SECONDARY
            cx = gx + gw // 2
            self._fill((cx - S(4), gy + gh, cx + S(4), gy + gh + S(4)), 0, stand)
            self._fill((cx - S(10), gy + gh + S(4), cx + S(10), gy + gh + S(6)), S(1), stand)
            self._draw.text((gx + gw / 2, gy + gh / 2), slot.value.title(), font=fl,
                            fill=label, anchor="mm")

        ty = gy + gh + S(17)
        self._fill((centers[0], ty - S(1), centers[2], ty + S(1)), S(1), (255, 255, 255, 50))
        if active:
            span = centers[1] - centers[0]
            reach = max(1e-3, v.side_threshold * 1.6)
            dx = max(-1.0, min(1.0, v.x / reach)) * span
            r = S(5)
            self._circle(centers[1] + dx, ty, r + 1, (0, 0, 0, 70))
            self._circle(centers[1] + dx, ty, r, LABEL)

    # ----------------------------------------------------------- calibration
    def _render_calibration(self, frame_bgr: np.ndarray, v: PreviewView) -> np.ndarray:
        """Face ID-style enrollment screen: black background, the face in a
        circle, a ring of ticks that turn green as the head covers each
        direction, and one instruction underneath."""
        h0, w0 = frame_bgr.shape[:2]
        W = self.width
        H = max(1, int(round(h0 * W / w0)))
        self._s = W / _BASE_WIDTH
        self._W, self._H = W, H
        S = self._px
        self._base = Image.new("RGB", (W, H), (0, 0, 0))
        self._layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._layer)

        r = int(H * 0.27)
        cx, cy = W / 2, H * 0.43
        crop = self._face_crop(frame_bgr, v.landmarks if v.face else None, 2 * r)
        if not v.face or not v.calib_collecting:
            crop = cv2.convertScaleAbs(crop, alpha=0.6 if v.face else 0.35)
        disc = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self._base.paste(disc, (int(cx - r), int(cy - r)), self._mask(2 * r, 2 * r, r))
        outer = self._ticks(cx, cy, r, v)

        # Step row: done = green, current = white, upcoming = tertiary.
        fs = self._font(S(13), 600)
        names = [s.title() for s in v.calib_targets] or ["Center", "Left", "Right"]
        gap = S(22)
        widths = [self._draw.textlength(t, font=fs) for t in names]
        x = (W - sum(widths) - gap * (len(names) - 1)) / 2
        for i, (name, w) in enumerate(zip(names, widths)):
            color = (GREEN + (255,) if i < v.calib_step
                     else LABEL if i == v.calib_step else TERTIARY)
            self._draw.text((x, S(26)), name, font=fs, fill=color, anchor="ls")
            x += w + gap

        target = (v.calib_target or "center").lower()
        ty = cy + outer + S(8)
        self._draw.text((W / 2, ty), f"Look at your {target} monitor",
                        font=self._font(S(20), 600), fill=LABEL, anchor="mt")
        if not v.face:
            body = "Can't see your face. Face the camera."
        elif v.calib_flash:
            body = "Done"
        elif v.calib_collecting:
            body = "Keep your eyes on it and slowly move your head to complete the circle."
        else:
            body = f"Move your eyes there. Starting in {max(1, math.ceil(v.calib_countdown))}"
        fb = self._font(S(13), 400)
        self._draw.text((W / 2, ty + S(30)), self._ellipsize(body, fb, W - S(40)),
                        font=fb, fill=SECONDARY, anchor="mt")

        out = Image.alpha_composite(self._base.convert("RGBA"), self._layer).convert("RGB")
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

    def _face_crop(self, frame: np.ndarray, lm: Optional[np.ndarray], size: int) -> np.ndarray:
        """Square crop that keeps the face centred and filling the circle,
        smoothed so it doesn't jitter with the landmarks."""
        h0, w0 = frame.shape[:2]
        if lm is not None:
            fx = lm[1, 0] * w0
            fy = (lm[10, 1] + lm[152, 1]) / 2 * h0
            side = 1.7 * max(abs(lm[454, 0] - lm[234, 0]) * w0, abs(lm[152, 1] - lm[10, 1]) * h0)
            target = (fx, fy, max(side, 40.0))
        else:
            target = self._crop or (w0 / 2, h0 / 2, h0 * 0.9)
        if self._crop is None:
            self._crop = target
        else:
            self._crop = tuple(0.75 * a + 0.25 * b for a, b in zip(self._crop, target))
        fx, fy, side = self._crop
        scale = size / side
        m = np.float32([[scale, 0, size / 2 - fx * scale], [0, scale, size / 2 - fy * scale]])
        return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _ticks(self, cx: float, cy: float, r: int, v: PreviewView) -> int:
        """Ring of radial ticks around the face circle. Returns its outer radius."""
        S = self._px
        n, k = 72, 2  # ticks; supersampling factor for anti-aliasing
        inner, short, long = r + S(10), S(11), S(19)
        outer = inner + long + S(2)
        size = 2 * outer * k
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c = size / 2
        cov = v.calib_coverage
        width = max(2, S(3)) * k
        for i in range(n):
            lit = v.calib_flash or (v.calib_collecting and bool(cov) and cov[i * len(cov) // n])
            length = long if lit else short
            color = GREEN + (255,) if lit else (255, 255, 255, 95 if v.calib_collecting else 45)
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            d.line((c + ca * inner * k, c + sa * inner * k,
                    c + ca * (inner + length) * k, c + sa * (inner + length) * k),
                   fill=color, width=width)
        self._layer.alpha_composite(img.reduce(k), (int(cx - outer), int(cy - outer)))
        return outer

    def _face(self, v: PreviewView) -> None:
        lm = v.landmarks
        S = self._px
        W, H = self._W, self._H
        if v.details:
            for idx in (33, 133, 263, 362, 1, 10, 152):
                self._circle(lm[idx, 0] * W, lm[idx, 1] * H, S(2), (255, 255, 255, 120))
        for idx in (468, 473):  # iris centres
            x, y = lm[idx, 0] * W, lm[idx, 1] * H
            self._circle(x, y, S(4), (0, 0, 0, 80))
            self._circle(x, y, S(3), (255, 255, 255, 235))

    # ------------------------------------------------------------ primitives
    def _px(self, value: float) -> int:
        return max(1, int(round(value * self._s)))

    def _font(self, size: int, weight: int = 400, mono: bool = False):
        key = (size, weight, mono)
        font = self._fonts.get(key)
        if font is None:
            font = self._load_font(size, weight, mono)
            self._fonts[key] = font
        return font

    @staticmethod
    def _load_font(size: int, weight: int, mono: bool):
        if mono:
            for name in ("CascadiaMono.ttf", "consola.ttf"):
                path = _FONT_DIR / name
                if path.exists():
                    return ImageFont.truetype(str(path), size)
        else:
            variable = _FONT_DIR / "SegUIVar.ttf"
            if variable.exists():
                font = ImageFont.truetype(str(variable), size)
                try:
                    # Optical size axis is in points: px * 0.75 at 96 dpi.
                    font.set_variation_by_axes([weight, max(5.0, min(36.0, size * 0.75))])
                except (OSError, ValueError):
                    pass
                return font
            static = _FONT_DIR / ("seguisb.ttf" if weight >= 600 else "segoeui.ttf")
            if static.exists():
                return ImageFont.truetype(str(static), size)
        return ImageFont.load_default(size)

    def _ellipsize(self, text: str, font, max_width: float) -> str:
        if self._draw.textlength(text, font=font) <= max_width:
            return text
        while text and self._draw.textlength(text + "…", font=font) > max_width:
            text = text[:-1]
        return text.rstrip() + "…"

    def _mask(self, w: int, h: int, r: int, outline: int = 0) -> Image.Image:
        """Anti-aliased rounded-rect mask (or ring when outline > 0)."""
        key = (w, h, r, outline)
        mask = self._masks.get(key)
        if mask is None:
            k = 4
            big = Image.new("L", (w * k, h * k), 0)
            d = ImageDraw.Draw(big)
            rect = [0, 0, w * k - 1, h * k - 1]
            if outline:
                d.rounded_rectangle(rect, radius=r * k, outline=255, width=outline * k)
            else:
                d.rounded_rectangle(rect, radius=r * k, fill=255)
            mask = big.resize((w, h), Image.LANCZOS)
            if len(self._masks) > 256:
                self._masks.clear()
            self._masks[key] = mask
        return mask

    def _shape(self, box, r: int, rgba, outline: int = 0) -> None:
        x0, y0, x1, y1 = (int(round(c)) for c in box)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        r = min(r, w // 2, h // 2)
        key = (w, h, r, outline, tuple(rgba))
        img = self._fills.get(key)
        if img is None:
            mask = self._mask(w, h, r, outline)
            if rgba[3] < 255:
                mask = mask.point(lambda value, a=rgba[3]: value * a // 255)
            img = Image.new("RGBA", (w, h), tuple(rgba[:3]) + (0,))
            img.putalpha(mask)
            if len(self._fills) > 256:
                self._fills.clear()
            self._fills[key] = img
        self._layer.alpha_composite(img, (x0, y0))

    def _fill(self, box, r: int, rgba) -> None:
        self._shape(box, r, rgba)

    def _ring(self, box, r: int, rgba, width: int) -> None:
        self._shape(box, r, rgba, outline=max(1, width))

    def _polyline(self, points, rgba, width: int) -> None:
        """Anti-aliased open polyline (chevrons, checkmarks)."""
        k = 4
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0 = int(min(xs)) - width - 1, int(min(ys)) - width - 1
        w, h = int(max(xs)) - x0 + width + 2, int(max(ys)) - y0 + width + 2
        img = Image.new("RGBA", (w, h), tuple(rgba[:3]) + (0,))
        mask = Image.new("L", (w * k, h * k), 0)
        d = ImageDraw.Draw(mask)
        pts = [((x - x0) * k, (y - y0) * k) for x, y in points]
        d.line(pts, fill=rgba[3], width=width * k, joint="curve")
        r = width * k / 2
        for x, y in (pts[0], pts[-1]):
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgba[3])
        img.putalpha(mask.resize((w, h), Image.LANCZOS))
        self._layer.alpha_composite(img, (max(0, x0), max(0, y0)))

    def _circle(self, cx: float, cy: float, r: int, rgba) -> None:
        self._shape((cx - r, cy - r, cx + r, cy + r), r, rgba)

    def _glass(self, box, r: int) -> None:
        """Blurred backdrop + tint + light edge."""
        x0, y0, x1, y1 = (int(round(c)) for c in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self._W, x1), min(self._H, y1)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        r = min(r, w // 2, h // 2)
        self._base.paste(self._blur.crop((x0, y0, x1, y1)), (x0, y0), self._mask(w, h, r))
        self._fill((x0, y0, x1, y1), r, GLASS_TINT)
        self._ring((x0, y0, x1, y1), r, GLASS_EDGE, 1)
