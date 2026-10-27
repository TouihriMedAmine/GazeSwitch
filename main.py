"""GazeSwitch - webcam-based multi-monitor targeting system.

Run from the project root:
    python main.py                     # observe-only mode (default, safe)
    python main.py --list-cameras      # find the Camo camera index
    python main.py --list-monitors     # verify monitor->slot mapping
    python main.py --probe             # print live feature values for tuning
    python main.py --enable-targeting  # allow arming targeting at runtime
    python main.py --tray              # desktop app: tray icon, no console needed

Preview-window keys: C calibrate | T toggle targeting (if gated on) |
E emergency-off | D details | N next camera | S silent mode | Q/ESC quit
(tray mode: hide preview); click the camera name to pick a camera. Global
hotkeys (configurable in config.yaml): Ctrl+Alt+G switching, Ctrl+Alt+K
emergency-off, Ctrl+Alt+P preview, Ctrl+Alt+S silent, Ctrl+Alt+C calibrate,
Ctrl+Alt+N next camera, Ctrl+Alt+Q quit. The packaged GazeSwitch.exe always
runs in tray mode.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src import app_env
from src.action_router import ActionRouter, TransitionLogAction
from src import sound
from src.calibration import MonitorCalibration, CalibrationError, load_baseline, save_baseline
from src.camera import CameraCapture
from src.camera_devices import CameraDevice, find_camera, list_cameras
from src.config import AppConfig, load_config, set_config_value
from src.gaze_features import FeatureExtractorConfig, GazeFeatureExtractor
from src.input_hotkeys import HotkeyMonitor, format_combo
from src.monitor_classifier import ClassifierConfig, MonitorClassifier, MonitorState
from src.overlay import PreviewRenderer, PreviewView, style_window
from src.selection_state import SelectionConfig, SelectionStateMachine
from src.target_manager import MonitorTargetRegistry, WindowPreference
from src.targeting import TargetingConfig as TargetingRuntimeConfig
from src.targeting import TargetingController
from src.tray import TrayController
from src.window_manager import WindowManager, map_monitors_to_slots

log = logging.getLogger("gazeswitch.main")

# Global shortcuts; each name is also the request name used by the tray menu
# and the preview, and the key in config.yaml's hotkeys section.
HOTKEY_ACTIONS = ("toggle_targeting", "emergency_disable", "toggle_preview", "toggle_silent",
                  "calibrate", "next_camera", "quit")


# --------------------------------------------------------------------- setup
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="GazeSwitch",
        description="Webcam-based multi-monitor targeting (observe-only by default).",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument(
        "--no-preview", action="store_true", help="Headless mode: console status only"
    )
    parser.add_argument(
        "--enable-targeting",
        action="store_true",
        help="Allow arming the (Phase 2) focus-targeting mode at runtime. "
        "Without this flag the app stays in observe-only mode forever.",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Run as a desktop app with a system-tray icon. Targeting can be "
        "armed whenever config.targeting.enabled is true (no --enable-targeting needed).",
    )
    parser.add_argument(
        "--probe", action="store_true", help="Print live feature values for threshold tuning"
    )
    parser.add_argument("--list-cameras", action="store_true", help="List camera indices and exit")
    parser.add_argument("--list-monitors", action="store_true", help="List monitors/windows and exit")
    parser.add_argument("--log-level", default=None, help="Override log level (e.g. DEBUG)")
    return parser.parse_args(argv)


def camera_candidates(
    name: Optional[str], index: Optional[int], backend: str, devices: list[CameraDevice]
) -> list[tuple[int, str, str]]:
    """(index, backend, label) to try in order: the named device, else the
    configured index, else every detected camera (scan 0..5 without names)."""
    if name:
        device = find_camera(name, devices)
        if device is not None:
            return [(device.index, device.backend, device.name)]
        log.warning("Camera '%s' not found - using the first working camera", name)
    if index is not None:
        label = next((d.name for d in devices if d.index == index and d.backend == backend),
                     f"Camera {index}")
        return [(index, backend, label)]
    if devices:
        return [(d.index, d.backend, d.name) for d in devices]
    return [(i, backend, f"Camera {i}") for i in range(6)]


# ------------------------------------------------------------------- helpers
def build_preferences(config: AppConfig) -> dict:
    prefs: dict = {}
    for slot, entries in config.windows.preferences.items():
        prefs[slot] = [WindowPreference(**e) for e in entries if isinstance(e, dict)]
    return prefs


# ----------------------------------------------------------------------- app
class GazeSwitchApp:
    def __init__(self, config: AppConfig, args: argparse.Namespace) -> None:
        self.cfg = config
        self.args = args
        self.tray: Optional[TrayController] = TrayController() if args.tray else None
        self.enabled_gate = bool(
            config.targeting.enabled and (args.enable_targeting or args.tray)
        )
        if args.enable_targeting and not config.targeting.enabled:
            log.warning(
                "--enable-targeting given but config.targeting.enabled is false; "
                "set enabled: true in config.yaml to allow arming."
            )

        self.wm = WindowManager(excluded_processes=config.windows.excluded_processes)
        self.dpi_mode = self.wm.set_process_dpi_awareness()
        log.info("DPI awareness: %s", self.dpi_mode)

        monitors = self.wm.enumerate_monitors()
        self.slots, self.slot_note = map_monitors_to_slots(monitors)
        log.info("Monitor mapping: %s", self.slot_note)
        for slot, mon in self.slots.items():
            log.info(
                "  %-6s -> %s %s rect=%s dpi=%d",
                slot,
                mon.device,
                "(primary)" if mon.primary else "",
                mon.rect,
                mon.dpi,
            )

        self.registry = MonitorTargetRegistry(
            self.wm,
            refresh_s=config.windows.refresh_s,
            preferences=build_preferences(config),
        )

        self.classifier = MonitorClassifier(
            ClassifierConfig(
                center_threshold=config.classifier.center_threshold,
                side_threshold=config.classifier.side_threshold,
                invert_x=config.classifier.invert_x,
                auto_thresholds_from_noise=config.classifier.auto_thresholds_from_noise,
                monitor_band=config.classifier.monitor_band,
            )
        )
        self.extractor = GazeFeatureExtractor(
            FeatureExtractorConfig(
                weight_head=config.classifier.weight_head,
                weight_iris=config.classifier.weight_iris,
                yaw_scale=config.classifier.yaw_scale,
                iris_scale=config.classifier.iris_scale,
            )
        )
        self.state_machine = SelectionStateMachine(
            SelectionConfig(
                window_frames=config.selection.window_frames,
                min_confidence=config.selection.min_confidence,
                dwell_ms=config.selection.dwell_ms,
                lost_ms=config.selection.lost_ms,
                vote_window_ms=config.selection.vote_window_ms,
            )
        )
        self.router = ActionRouter()
        self.router.register(TransitionLogAction())

        self.targeting = TargetingController(
            TargetingRuntimeConfig(
                typing_quiet_ms=config.targeting.typing_quiet_ms,
                target_dwell_ms=config.targeting.target_dwell_ms,
                focus_backoff_s=config.targeting.focus_backoff_s,
            ),
            enabled_gate=self.enabled_gate,
        )

        feat = self.extractor.cfg
        wsum = feat.weight_head + feat.weight_iris
        self.calibration = MonitorCalibration(
            step_s=config.calibration.duration_s,
            settle_s=config.calibration.settle_s,
            min_samples=config.calibration.min_samples,
            max_step_s=config.calibration.max_duration_s,
            max_noise_x=config.calibration.max_noise_x,
            prior=(feat.weight_head / feat.yaw_scale / wsum,
                   feat.weight_iris / feat.iris_scale / wsum),
            monitor_band=config.classifier.monitor_band,
        )
        self.baseline = None
        self.calib_message: Optional[str] = None
        self.calib_message_until_ms = 0

        combos = {name: getattr(config.hotkeys, name) for name in HOTKEY_ACTIONS}
        combos = {name: combo for name, combo in combos.items() if combo and combo.strip()}
        self.hotkeys = HotkeyMonitor(combos)
        if self.tray is not None:
            self.tray.shortcuts = {name: format_combo(c) for name, c in combos.items()}
        self.silent = bool(config.overlay.silent)

        # Camera (created lazily in run(); tracker needs the model file first)
        self.camera: Optional[CameraCapture] = None
        self.camera_devices: list[CameraDevice] = []
        self.camera_choice = "" if args.camera is not None else (config.camera.name or "")
        self.camera_label = ""

        self.tracker = None
        self.last_sample = None  # latest GazeSample, for overlay diagnostics
        self.preview_window = "GazeSwitch"
        self.preview_visible = False
        self.preview_details = False
        self._preview_shape: Optional[tuple] = None
        self.renderer = PreviewRenderer(config.overlay.width)
        self.camera_menu_open = False
        self._pointer: Optional[tuple] = None
        self._clicks: list[tuple] = []
        self._preview_auto_opened = False
        self._last_probe_log_ms = 0
        self._last_status_log_ms = 0
        self._fps = 0.0

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        # Face tracker (may raise FaceTrackerError with actionable message)
        from src.face_tracker import FaceTracker, FaceTrackerError

        try:
            self.tracker = FaceTracker(
                model_path=app_env.resolve_resource(self.cfg.tracking.model_path),
                min_detection_confidence=self.cfg.tracking.min_detection_confidence,
                min_presence_confidence=self.cfg.tracking.min_presence_confidence,
                min_tracking_confidence=self.cfg.tracking.min_tracking_confidence,
                head_pose_method=self.cfg.tracking.head_pose_method,
                blink_threshold=self.cfg.tracking.blink_threshold,
                invert_yaw=self.cfg.tracking.invert_yaw,
                invert_pitch=self.cfg.tracking.invert_pitch,
                blendshapes=self.cfg.tracking.blendshapes,
            )
        except FaceTrackerError as exc:
            self._fatal(str(exc))
            return 2

        # Baseline auto-load
        if self.cfg.calibration.auto_load:
            try:
                baseline = load_baseline(self.cfg.calibration.save_path)
                self.baseline = baseline
                self.classifier.set_baseline(baseline)
                log.info("Reusing saved baseline from %s (press C to recalibrate any time)",
                         self.cfg.calibration.save_path)
                if baseline.model is None:
                    self.calib_message = ("Press C to recalibrate: GazeSwitch now learns how "
                                          "your head and eyes move together")
                    self.calib_message_until_ms = int(time.monotonic() * 1000) + 15000
            except CalibrationError:
                log.info("No usable saved baseline - press C in the preview to calibrate "
                         "(center, left, then right monitor).")

        # Camera
        self.camera_devices = list_cameras()
        index = self.args.camera if self.args.camera is not None else self.cfg.camera.index
        if not self._open_camera(self.camera_choice or None, index):
            self._fatal("No working camera found. Make sure your camera (or Camo, Iriun, "
                        "DroidCam...) is running, then try: python main.py --list-cameras")
            return 3

        self.hotkeys.start()
        if self.tray is not None:
            self.tray.start()
            if self.enabled_gate and self.cfg.targeting.arm_on_start:
                self.targeting.toggle()
            if not self.classifier.calibrated:
                # First run: show the camera so the user can calibrate.
                self._set_preview(True)
                self._preview_auto_opened = True
                self._notify("To start, press C in the preview. You'll look at your "
                             "center, left and right monitors.")
            elif not self.classifier.learned:
                self._notify("Recalibrate to use your head and eye movement together: "
                             "right-click the eye icon and choose Calibrate.")
            else:
                preview = self.tray.shortcuts.get("toggle_preview")
                self._notify("GazeSwitch is running. Right-click the eye icon near the clock "
                             "for options" + (f", or press {preview} for the preview." if preview
                                              else "."))
        elif self.cfg.overlay.show and not self.args.no_preview:
            self._set_preview(True)

        log.info("GazeSwitch running in %s mode. Press C to calibrate, Q to quit.",
                 "ARMABLE-targeting" if self.enabled_gate else "observe-only")

        exit_code = 0
        try:
            exit_code = self._loop()
        except KeyboardInterrupt:
            log.info("Interrupted by user (Ctrl+C)")
        finally:
            self.shutdown()
        return exit_code

    def _fatal(self, message: str) -> None:
        log.error("%s", message)
        if self.tray is not None or sys.stderr is None:
            app_env.message_box(message, error=True)

    def _open_camera(self, name: Optional[str], index: Optional[int] = None) -> bool:
        """Open the named camera (or the first working one) as self.camera."""
        for idx, backend, label in camera_candidates(
            name, index, self.cfg.camera.backend, self.camera_devices
        ):
            camera = CameraCapture(
                index=idx,
                width=self.cfg.camera.width,
                height=self.cfg.camera.height,
                fps=self.cfg.camera.fps,
                mirror=self.cfg.camera.mirror,
                process_width=self.cfg.camera.process_width,
                backend=backend,
                max_read_failures=self.cfg.camera.max_read_failures,
                reopen_delay_s=self.cfg.camera.reopen_delay_s,
            )
            if camera.open():
                self.camera, self.camera_label = camera, label
                log.info("Using camera '%s' (%s index %d)", label, backend, idx)
                return True
        return False

    def _switch_camera(self, name: str) -> None:
        """Runtime camera change from the tray. '' = automatic."""
        if name == self.camera_choice:
            return
        previous_choice = self.camera_choice
        if self.camera is not None:
            self.camera.release()
        if self._open_camera(name or None):
            self.camera_choice = name
            self.state_machine.reset()
            try:
                set_config_value(self.args.config, "camera", "name", name or None)
            except OSError as exc:
                log.warning("Could not save the camera choice: %s", exc)
            message = f"Using {self.camera_label}"
            if self.classifier.calibrated:
                message += ". Recalibrate if this camera sits in a different place."
        else:
            self._open_camera(previous_choice or None)
            message = f"Couldn't open {name}. Still using {self.camera_label}."
        now_ms = int(time.monotonic() * 1000)
        self._show_message(message, now_ms, notify=True)

    def _set_preview(self, visible: bool) -> None:
        if visible and not self.preview_visible:
            cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.preview_window, self.cfg.overlay.width,
                int(self.cfg.overlay.width * 9 / 16),
            )
            self._preview_shape = None
            try:
                cv2.moveWindow(self.preview_window, *self.cfg.overlay.position)
            except Exception:  # noqa: BLE001
                pass
            style_window(self.preview_window, topmost=self.cfg.overlay.always_on_top)
            cv2.setMouseCallback(self.preview_window, self._on_mouse)
            self._pointer, self._clicks = None, []
        elif not visible and self.preview_visible:
            self.camera_menu_open = False
            try:
                cv2.destroyWindow(self.preview_window)
            except cv2.error:
                pass
        self.preview_visible = visible
        if not visible:
            self._preview_auto_opened = False

    def _start_calibration(self, now_ms: int) -> None:
        if self.calibration.running:
            return
        self.calibration.start(now_ms)
        self.calib_message = None
        self.camera_menu_open = False
        if self.silent and not self.preview_visible:
            # No beeps in silent mode: the preview has to guide instead.
            self._set_preview(True)
            self._preview_auto_opened = True
        self._notify("Calibrating: look at your center, then left, then right monitor. "
                     "At each beep, keep your eyes on it and move your head around.")

    def _notify(self, message: str, force: bool = False) -> None:
        """Tray notification, unless silent mode is on (force: confirmations
        of the user's own shortcut that would otherwise be invisible)."""
        if self.tray is not None and (force or not self.silent):
            self.tray.notify(message)

    def _show_message(self, message: str, now_ms: int, notify: bool = False) -> None:
        self.calib_message = message
        self.calib_message_until_ms = now_ms + 4000
        if notify and not self.preview_visible:
            self._notify(message)

    def _handle_request(self, request: str, now_ms: int) -> bool:
        """Apply one action from the tray menu, a global shortcut or the
        preview. Returns True when the user quit."""
        if request == "quit":
            log.info("Quit requested")
            return True
        if request == "toggle_targeting":
            self._show_message(self.targeting.toggle(), now_ms, notify=True)
        elif request == "emergency_disable":
            self._show_message(self.targeting.emergency_off(), now_ms, notify=True)
        elif request == "calibrate":
            self._start_calibration(now_ms)
        elif request == "toggle_preview":
            self._set_preview(not self.preview_visible)
        elif request == "toggle_silent":
            self._toggle_silent(now_ms)
        elif request == "next_camera":
            self._switch_camera(self._next_camera_choice())
        elif request.startswith("camera:"):
            self._switch_camera(request.removeprefix("camera:"))
        elif request == "refresh_cameras":
            self.camera_devices = list_cameras()
        return False

    def _handle_tray_requests(self, now_ms: int) -> bool:
        """Apply queued tray-menu requests. Returns True when the user quit."""
        if self.tray is None:
            return False
        while (request := self.tray.poll()) is not None:
            if self._handle_request(request, now_ms):
                return True
        return False

    def _toggle_silent(self, now_ms: int) -> None:
        self.silent = not self.silent
        try:
            set_config_value(self.args.config, "overlay", "silent", self.silent)
        except OSError as exc:
            log.warning("Could not save silent mode: %s", exc)
        message = ("Silent mode on. No notifications or beeps." if self.silent
                   else "Silent mode off")
        log.info(message)
        self.calib_message, self.calib_message_until_ms = message, now_ms + 4000
        if not self.preview_visible:
            self._notify(message, force=True)

    def _next_camera_choice(self) -> str:
        """Automatic -> each camera -> Automatic."""
        options = [""] + [d.name for d in self.camera_devices]
        try:
            return options[(options.index(self.camera_choice) + 1) % len(options)]
        except ValueError:
            return ""

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ARG002 - OpenCV signature
        self._pointer = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self._clicks.append((x, y))

    def _handle_clicks(self, now_ms: int) -> bool:
        clicks, self._clicks = self._clicks, []
        for x, y in clicks:
            hit = self.renderer.hit_test(x, y)
            if hit == "camera_menu":
                self.camera_menu_open = not self.camera_menu_open
                if self.camera_menu_open:
                    self.camera_devices = list_cameras()
            elif hit == "refresh_cameras":
                self.camera_devices = list_cameras()
            elif hit is not None and hit.startswith("camera:"):
                self.camera_menu_open = False
                self._switch_camera(hit.removeprefix("camera:"))
            elif hit != "menu_body":
                self.camera_menu_open = False
        return False

    def _handle_preview_keys(self, now_ms: int) -> bool:
        """Pump the preview window. Returns True when the user quit."""
        key = cv2.waitKey(1) & 0xFF
        try:
            closed = cv2.getWindowProperty(self.preview_window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if key == 27 and self.camera_menu_open and not closed:
            self.camera_menu_open = False
            return False
        if key in (ord("q"), 27) or closed:
            if self.tray is None:
                return True
            self._set_preview(False)  # tray app keeps running; quit via the menu
            return False
        self._handle_clicks(now_ms)
        if key == ord("c"):
            self._start_calibration(now_ms)
        elif key == ord("t"):
            self._show_message(self.targeting.toggle(), now_ms)
        elif key == ord("e"):
            self._show_message(self.targeting.emergency_off(), now_ms)
        elif key == ord("d"):
            self.preview_details = not self.preview_details
        elif key == ord("n"):
            self._switch_camera(self._next_camera_choice())
        elif key == ord("s"):
            self._toggle_silent(now_ms)
        return False

    # ------------------------------------------------------------------ loop
    def _loop(self) -> int:
        assert self.camera is not None and self.tracker is not None
        last_frame_ts = time.monotonic()
        while True:
            now_ms = int(time.monotonic() * 1000)
            if self._handle_tray_requests(now_ms):
                return 0
            # Global shortcuts (handled even while the camera is down)
            for name in HOTKEY_ACTIONS:
                if self.hotkeys.consume(name) and self._handle_request(name, now_ms):
                    return 0

            ok, frame = self.camera.read()
            if not ok or frame is None:
                if self.preview_visible:
                    blank = np.full((360, 640, 3), 18, dtype=np.uint8)
                    self._show_canvas(self.renderer.render(blank, PreviewView(
                        camera_ok=False,
                        calibrated=self.classifier.calibrated,
                        armed=self.targeting.armed,
                        close_label="Hide" if self.tray is not None else "Quit",
                        **self._chrome_fields(),
                    )))
                    if self._handle_preview_keys(now_ms):
                        return 0
                time.sleep(0.05)
                continue

            # Frame timing / FPS
            now = time.monotonic()
            dt = now - last_frame_ts
            last_frame_ts = now
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if dt > 0 else self._fps

            # Detection on a downscaled copy for speed
            scale = min(1.0, self.cfg.camera.process_width / max(1, frame.shape[1]))
            if scale < 1.0:
                detect_frame = cv2.resize(
                    frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                )
            else:
                detect_frame = frame
            obs = self.tracker.detect(detect_frame, now_ms)
            tracking_ok = obs is not None

            sample = self.extractor.extract(obs, self.baseline, now_ms)
            self.last_sample = sample
            classification = self.classifier.classify(sample)
            event = self.state_machine.update(classification, tracking_ok, now_ms)

            # Calibration collection
            if self.calibration.running:
                self.calibration.add_sample(sample, now_ms)
                step_event = self.calibration.update(now_ms)
                if (step_event in ("collect", "step") and self.cfg.calibration.sounds
                        and not self.silent):
                    sound.cue(step_event)
                elif step_event == "done":
                    self._finish_calibration(now_ms)

            # Window / target refresh (throttled inside)
            self.registry.update(now_ms, self.slots)

            # Stable-transition actions (logging) and gated focus targeting
            self.router.update(event, lambda: self.registry.description(event.state))
            if not self.calibration.running:  # calibration looks around on purpose
                self.targeting.update(event, self.registry, now_ms)

            if self.tray is not None:
                self.tray.update_state(
                    monitor=event.state.value,
                    armed=self.targeting.armed,
                    gate=self.enabled_gate,
                    calibrated=self.classifier.calibrated,
                    preview=self.preview_visible,
                    calibrating=self.calibration.running,
                    cameras=tuple(d.name for d in self.camera_devices),
                    camera_choice=self.camera_choice,
                    camera_label=self.camera_label,
                    silent=self.silent,
                )

            # Probe mode: print feature stream for threshold tuning
            if self.args.probe and now_ms - self._last_probe_log_ms >= 200:
                self._last_probe_log_ms = now_ms
                print(
                    f"x={sample.x:+.3f} yaw={sample.yaw_rel:+.3f} "
                    f"iris={sample.iris_dx:+.3f} blink={sample.blink:.2f} "
                    f"cls={classification.state.value}"
                )

            # Periodic status log
            if now_ms - self._last_status_log_ms >= 5000:
                self._last_status_log_ms = now_ms
                state = event.state
                log.info(
                    "status: monitor=%s stable=%s target='%s' fps=%.1f cam=%s calibrated=%s",
                    state.value,
                    event.stable,
                    self.registry.description(state),
                    self._fps,
                    f"idx{self.camera.index}",
                    self.classifier.calibrated,
                )

            if self.preview_visible:
                self._render(frame, obs, classification, event)
                if self._handle_preview_keys(now_ms):
                    return 0

    def _render(self, frame, obs, classification, event) -> None:
        assert self.camera is not None
        now_ms = int(time.monotonic() * 1000)
        message = None
        if now_ms < self.calib_message_until_ms and self.calib_message:
            message = self.calib_message
        center_thr, side_thr = self.classifier._thresholds()
        sample = self.last_sample
        details: list[str] = []
        if self.preview_details:
            votes = "  ".join(f"{k.title()} {v}" for k, v in sorted(event.votes.items()))
            details = [
                f"x {classification.x:+.3f}  yaw {sample.yaw_rel if sample else 0.0:+.2f}"
                f"  iris {sample.iris_dx if sample else 0.0:+.3f}",
                f"center ≤ {center_thr:.2f}  side ≥ {side_thr:.2f}",
                f"votes  {votes or '-'}",
                f"{self._fps:.0f} fps  {self.camera_label} {self.camera.status.value}",
            ]
        view = PreviewView(
            state=event.state,
            stable=event.stable,
            calibrated=self.classifier.calibrated,
            face=obs is not None,
            armed=self.targeting.armed,
            target=self.registry.description(event.state),
            x=classification.x,
            center_threshold=center_thr,
            side_threshold=side_thr,
            landmarks=obs.landmarks if obs is not None else None,
            message=message,
            calibrating=self.calibration.running,
            calib_progress=self.calibration.progress(now_ms),
            calib_valid=self.calibration.n_valid,
            calib_needed=self.calibration.min_samples,
            calib_step=self.calibration.step_index,
            calib_steps=len(self.calibration.steps),
            calib_target=self.calibration.step.value if self.calibration.step else "",
            calib_collecting=self.calibration.collecting(now_ms),
            calib_targets=tuple(s.value for s in self.calibration.steps),
            calib_coverage=self.calibration.coverage,
            calib_countdown=self.calibration.settle_remaining(now_ms),
            calib_flash=self.calibration.just_completed_step(now_ms),
            details=self.preview_details,
            detail_lines=details,
            close_label="Hide" if self.tray is not None else "Quit",
            **self._chrome_fields(),
        )
        self._show_canvas(self.renderer.render(frame, view))

    def _chrome_fields(self) -> dict:
        """Preview fields for the top bar's camera picker and silent badge."""
        return dict(
            silent=self.silent,
            camera_name=self.camera_label,
            cameras=tuple(d.name for d in self.camera_devices),
            camera_choice=self.camera_choice,
            camera_menu=self.camera_menu_open,
            pointer=self._pointer,
        )

    def _show_canvas(self, canvas: np.ndarray) -> None:
        if canvas.shape[:2] != self._preview_shape:
            self._preview_shape = canvas.shape[:2]
            cv2.resizeWindow(self.preview_window, canvas.shape[1], canvas.shape[0])
        cv2.imshow(self.preview_window, canvas)

    def _finish_calibration(self, now_ms: int) -> None:
        result = self.calibration.finish()
        if result.ok and result.baseline is not None:
            self.baseline = result.baseline
            self.classifier.set_baseline(result.baseline)
            self.state_machine.reset()
            try:
                save_baseline(self.cfg.calibration.save_path, result.baseline)
            except OSError as exc:
                log.warning("Could not save baseline: %s", exc)
        else:
            log.warning(result.message)
        if self.cfg.calibration.sounds and not self.silent:
            sound.cue("done" if result.ok else "fail")
        self.calib_message = result.message
        self.calib_message_until_ms = now_ms + 6000
        if self.tray is not None:
            self._notify(result.message)
            if result.ok and self._preview_auto_opened:
                self._set_preview(False)

    # ------------------------------------------------------------------ misc
    def list_windows_report(self) -> None:
        monitors = self.wm.enumerate_monitors()
        slots, note = map_monitors_to_slots(monitors)
        print("Monitors:")
        for m in monitors:
            print(
                f"  handle=0x{m.handle:08X} device={m.device} primary={m.primary} "
                f"dpi={m.dpi} rect={m.rect}"
            )
        print(f"Slot mapping: {note}")
        for slot, mon in slots.items():
            print(f"  {slot}: {mon.device} rect={mon.rect}")
        windows = self.wm.enumerate_windows(monitors)
        print(f"\nVisible candidate windows ({len(windows)}):")
        for w in windows[:60]:
            areas = {k: v for k, v in w.areas.items() if v > 0}
            print(
                f"  z={w.z_order:3d} hwnd=0x{w.hwnd:08X} {w.process_name or '?':28s} "
                f"min={w.minimized} fg={w.foreground} areas={areas} title='{w.title[:60]}'"
            )

    def shutdown(self) -> None:
        log.info("Shutting down...")
        try:
            self.hotkeys.stop()
        except Exception:  # noqa: BLE001
            pass
        if self.tray is not None:
            self.tray.stop()
        if self.tracker is not None:
            self.tracker.close()
        if self.camera is not None:
            self.camera.release()
        if self.preview_visible:
            try:
                cv2.destroyWindow(self.preview_window)
            except Exception:  # noqa: BLE001
                pass
        log.info("Shutdown complete")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if app_env.is_frozen():
        args.tray = True
    if args.tray:
        # Relative paths in config.yaml (baselines, logs, model) resolve
        # against the app folder, whatever the launch directory is.
        os.chdir(app_env.app_dir())
        if not app_env.acquire_single_instance():
            app_env.message_box(
                "GazeSwitch is already running. Look for the eye icon in the "
                "system tray (you may need to click the ^ arrow near the clock)."
            )
            return 0
        if not os.path.isabs(args.config):
            app_env.ensure_user_config(args.config)

    config = load_config(args.config)
    from src.logger import setup_logging

    level = args.log_level or config.logging.level
    setup_logging(level=level, file_path=config.logging.file)

    app = GazeSwitchApp(config, args)
    if args.list_monitors:
        app.list_windows_report()
        return 0
    if args.list_cameras:
        from scripts.list_cameras import list_cameras

        list_cameras()
        return 0
    try:
        return app.run()
    except Exception as exc:  # noqa: BLE001 - windowless app: surface crashes
        log.exception("GazeSwitch crashed")
        if args.tray:
            app_env.message_box(f"GazeSwitch stopped because of an error:\n\n{exc}\n\n"
                                "Details are in the logs folder.", error=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
