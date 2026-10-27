"""Camera capture with reconnection handling.

Wraps cv2.VideoCapture with:
* Windows-friendly backend preference (DirectShow first, which is what Camo
  exposes, then MSMF, then default),
* automatic reconnection after repeated read failures,
* centralised mirroring so every downstream module sees one consistent
  frame convention (mirrored selfie view when ``mirror`` is enabled),
* a background reader thread that keeps only the newest frame, so the
  processing loop never blocks on the device or works on stale frames,
* an injectable frame source for tests (no physical camera required).
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

FrameSource = Callable[[], Tuple[bool, Optional[np.ndarray]]]


class CameraStatus(str, Enum):
    OK = "OK"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class CameraCapture:
    """Resilient camera reader.

    ``frame_source`` and ``open_fn`` are injection points for tests; when
    ``frame_source`` is provided the class never touches cv2.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
        max_read_failures: int = 30,
        reopen_delay_s: float = 2.0,
        frame_source: Optional[FrameSource] = None,
        open_fn: Optional[Callable[[Optional[int]], bool]] = None,
        process_width: int = 640,
        backend: str = "auto",
    ) -> None:
        self.index = index
        self.backend = backend.lower()
        self.request_size = (width, height)
        self.fps = fps
        self.mirror = mirror
        self.max_read_failures = max_read_failures
        self.reopen_delay_s = reopen_delay_s
        self.process_width = process_width
        self._frame_source = frame_source
        self._open_fn = open_fn

        self._cap: object = None
        self._consecutive_failures = 0
        self._last_reopen_attempt = 0.0
        self._opened_once = False
        self.status = CameraStatus.RECONNECTING

        self._reader: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_cond = threading.Condition()
        self._latest: Tuple[bool, Optional[np.ndarray]] = (False, None)
        self._latest_seq = 0
        self._consumed_seq = 0

    # ------------------------------------------------------------------ open
    def open(self) -> bool:
        if self._frame_source is not None:
            self._opened_once = True
            self.status = CameraStatus.OK
            return True
        if self._open_fn is not None:
            ok = bool(self._open_fn(self.index))
        else:
            ok = self._open_cv(self.index)
        if ok:
            self._opened_once = True
            self._consecutive_failures = 0
            self.status = CameraStatus.OK
            log.info("Camera opened (index=%s)", self.index)
        else:
            self.status = CameraStatus.FAILED
            log.error("Could not open camera index=%s", self.index)
        return ok

    def _open_cv(self, index: Optional[int]) -> bool:
        import cv2  # local import keeps module importable in headless tests

        backends = {
            "msmf": [cv2.CAP_MSMF],
            "dshow": [cv2.CAP_DSHOW],
            # Media Foundation first: with Camo at 4K it delivers ~30 fps where
            # DirectShow manages ~9.
            "auto": [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY],
        }.get(self.backend, [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY])
        candidates: list[tuple[int, Optional[int]]] = []
        if index is not None:
            candidates = [(index, backend) for backend in backends]
        for idx, backend in candidates:
            cap = cv2.VideoCapture(idx, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.request_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.request_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Confirm the device actually delivers frames before trusting it.
            ok, frame = cap.read()
            if cap.isOpened() and ok and frame is not None:
                self._cap = cap
                log.info("Camera backend %s delivers %dx%d", cap.getBackendName(),
                         frame.shape[1], frame.shape[0])
                self._start_reader()
                return True
            cap.release()
        return False

    def _start_reader(self) -> None:
        self._reader_stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name="camera-reader", daemon=True
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        import cv2

        cap = self._cap
        while cap is not None and not self._reader_stop.is_set():
            try:
                ok, frame = cap.read()  # type: ignore[union-attr]
            except cv2.error as exc:
                log.warning("cv2 read error: %s", exc)
                ok, frame = False, None
            if ok and frame is not None:
                frame = self._resize(frame)
            with self._frame_cond:
                self._latest = (bool(ok), frame if ok else None)
                self._latest_seq += 1
                self._frame_cond.notify_all()
            if not ok:
                time.sleep(0.01)

    # ------------------------------------------------------------------ read
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame. Never raises; returns (False, None) on failure."""
        if self._frame_source is not None:
            try:
                ok, frame = self._frame_source()
            except Exception as exc:  # noqa: BLE001 - a failing source is a failed read
                log.warning("Frame source raised: %s", exc)
                ok, frame = False, None
        elif self._cap is None:
            ok, frame = False, None
        else:
            ok, frame = self._read_cv()

        if not ok or frame is None:
            self._consecutive_failures += 1
            self._handle_failure()
            return False, None

        if self._consecutive_failures >= self.max_read_failures:
            log.info("Camera recovered after %d failed reads", self._consecutive_failures)
        self._consecutive_failures = 0
        if self.status is not CameraStatus.OK:
            self.status = CameraStatus.OK

        # Some virtual cameras (e.g. Camo at 4K) ignore size negotiation.
        # Downscale FIRST (one pass over the big frame), then mirror the
        # small one - much cheaper than mirroring 4K.
        frame = self._resize(frame)

        if self.mirror:
            frame = frame[:, ::-1, :].copy()  # horizontal flip, contiguous
        return True, frame

    def _read_cv(self, timeout_s: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the newest frame from the reader thread, waiting briefly
        for one that has not been handed out yet."""
        with self._frame_cond:
            if self._latest_seq == self._consumed_seq:
                self._frame_cond.wait(timeout=timeout_s)
            if self._latest_seq == self._consumed_seq:
                return False, None
            self._consumed_seq = self._latest_seq
            return self._latest

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        pw = self.process_width
        if pw and frame.shape[1] > pw:
            h = max(1, int(round(frame.shape[0] * pw / frame.shape[1])))
            return cv2.resize(frame, (pw, h), interpolation=cv2.INTER_AREA)
        return frame

    def _handle_failure(self) -> None:
        if self._frame_source is not None:
            if self._consecutive_failures >= self.max_read_failures:
                self.status = CameraStatus.FAILED
            elif self._consecutive_failures > 0:
                self.status = CameraStatus.RECONNECTING
            return
        if self._consecutive_failures >= self.max_read_failures:
            self.status = CameraStatus.FAILED
            now = time.monotonic()
            if now - self._last_reopen_attempt >= self.reopen_delay_s:
                log.warning(
                    "%d consecutive camera failures; attempting to reopen",
                    self._consecutive_failures,
                )
                self._last_reopen_attempt = now
                self._close()
                self.open()

    # --------------------------------------------------------------- context
    def resolution(self) -> Tuple[int, int]:
        if self._frame_source is not None:
            return 0, 0
        if self._cap is None:
            return 0, 0
        import cv2

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[union-attr]
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[union-attr]
        return w, h

    def _close(self) -> None:
        if self._reader is not None:
            self._reader_stop.set()
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._cap is not None:
            try:
                self._cap.release()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - release must never crash shutdown
                pass
            self._cap = None

    def release(self) -> None:
        self._close()
        log.info("Camera released")

    @property
    def opened_once(self) -> bool:
        return self._opened_once
