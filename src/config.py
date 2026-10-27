"""Configuration loading (config.yaml) with typed defaults.

Every setting has a default; a partial config.yaml is merged over the
defaults. Unknown keys produce a warning instead of failing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: Optional[int] = None  # None = auto-scan 0..5
    name: Optional[str] = None  # device name; wins over index when set
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    process_width: int = 640
    backend: str = "auto"  # auto (MSMF, then DirectShow) | msmf | dshow
    max_read_failures: int = 30
    reopen_delay_s: float = 2.0


@dataclass
class TrackingConfig:
    model_path: str = "models/face_landmarker.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    head_pose_method: str = "geometric"  # geometric | matrix
    blink_threshold: float = 0.5
    invert_yaw: bool = False
    invert_pitch: bool = False
    blendshapes: bool = False


@dataclass
class CalibrationConfig:
    duration_s: float = 8.0  # collection time per monitor (ends early once the circle is full)
    settle_s: float = 3.0  # time to move your eyes before each monitor's collection
    min_samples: int = 40  # per monitor
    max_noise_x: float = 0.12
    max_duration_s: float = 14.0  # per monitor, when frames are being dropped
    sounds: bool = True
    save_path: str = "baselines/center_baseline.json"
    auto_load: bool = True


@dataclass
class ClassifierSection:
    invert_x: bool = False
    weight_head: float = 0.65
    weight_iris: float = 0.35
    yaw_scale: float = 0.45
    iris_scale: float = 0.18
    center_threshold: float = 0.12
    side_threshold: float = 0.32
    auto_thresholds_from_noise: bool = False
    monitor_band: float = 0.1


@dataclass
class SelectionSection:
    window_frames: int = 7
    min_confidence: float = 0.6
    dwell_ms: int = 350
    lost_ms: int = 600
    vote_window_ms: int = 0


@dataclass
class WindowsConfig:
    refresh_s: float = 2.0
    excluded_processes: List[str] = field(default_factory=lambda: ["camo.exe", "camo studio"])
    preferences: Dict[str, List[Dict[str, str]]] = field(
        default_factory=lambda: {"LEFT": [], "CENTER": [], "RIGHT": []}
    )


@dataclass
class TargetingConfig:
    enabled: bool = False  # master gate; --enable-targeting also required
    typing_quiet_ms: int = 700
    target_dwell_ms: int = 800
    focus_backoff_s: float = 1.3
    arm_on_start: bool = False


@dataclass
class HotkeysConfig:
    toggle_targeting: str = "ctrl+alt+g"
    emergency_disable: str = "ctrl+alt+k"
    toggle_preview: str = "ctrl+alt+p"
    toggle_silent: str = "ctrl+alt+s"
    calibrate: str = "ctrl+alt+c"
    next_camera: str = "ctrl+alt+n"
    quit: str = "ctrl+alt+q"


@dataclass
class OverlayConfig:
    show: bool = True
    width: int = 720
    position: List[int] = field(default_factory=lambda: [40, 40])
    always_on_top: bool = True
    silent: bool = False  # no notifications or beeps (toggle: hotkeys.toggle_silent)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/gazeswitch.log"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    classifier: ClassifierSection = field(default_factory=ClassifierSection)
    selection: SelectionSection = field(default_factory=SelectionSection)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    targeting: TargetingConfig = field(default_factory=TargetingConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build_section(cls: type, data: Dict[str, Any], section: str) -> Any:
    """Instantiate a dataclass section from a dict, warning on unknown keys."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        log.warning("Unknown config keys in [%s]: %s (ignored)", section, sorted(unknown))
    kwargs = {k: v for k, v in data.items() if k in known}
    try:
        return cls(**kwargs)  # type: ignore[call-arg]
    except TypeError as exc:
        log.error("Invalid config in [%s]: %s - using defaults for this section", section, exc)
        return cls()  # type: ignore[call-arg]


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config.yaml from disk; missing file -> pure defaults."""
    cfg = AppConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        log.warning("Config file %s not found - using built-in defaults", p)
        return cfg
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log.error("Config file %s is not valid YAML (%s) - using defaults", p, exc)
        return cfg
    if not isinstance(data, dict):
        log.error("Config file %s must be a YAML mapping - using defaults", p)
        return cfg

    section_map = {
        "camera": (CameraConfig, "camera"),
        "tracking": (TrackingConfig, "tracking"),
        "calibration": (CalibrationConfig, "calibration"),
        "classifier": (ClassifierSection, "classifier"),
        "selection": (SelectionSection, "selection"),
        "windows": (WindowsConfig, "windows"),
        "targeting": (TargetingConfig, "targeting"),
        "hotkeys": (HotkeysConfig, "hotkeys"),
        "overlay": (OverlayConfig, "overlay"),
        "logging": (LoggingConfig, "logging"),
    }
    for key, (cls, attr) in section_map.items():
        section_data = data.get(key)
        if section_data is None:
            continue
        if not isinstance(section_data, dict):
            log.warning("Config section [%s] must be a mapping - ignored", key)
            continue
        setattr(cfg, attr, _build_section(cls, section_data, key))

    unknown_sections = set(data) - set(section_map)
    if unknown_sections:
        log.warning("Unknown config sections: %s (ignored)", sorted(unknown_sections))
    return cfg


def set_config_value(path: str, section: str, key: str, value: Any) -> None:
    """Write one ``section.key`` value into config.yaml in place, keeping
    every other line (and comment) untouched."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True) if p.exists() else []
    rendered = "null" if value is None else json.dumps(value)  # JSON scalars are valid YAML
    section_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    key_re = re.compile(rf"^(\s+){re.escape(key)}:")

    start = next((i for i, line in enumerate(lines) if section_re.match(line.rstrip("\r\n"))), None)
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += [f"\n{section}:\n", f"  {key}: {rendered}\n"]
    else:
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
            end += 1
        for i in range(start + 1, end):
            match = key_re.match(lines[i])
            if match:
                lines[i] = f"{match.group(1)}{key}: {rendered}\n"
                break
        else:
            lines.insert(start + 1, f"  {key}: {rendered}\n")
    p.write_text("".join(lines), encoding="utf-8")
