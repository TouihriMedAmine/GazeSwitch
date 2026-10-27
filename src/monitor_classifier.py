"""Per-frame gaze classification into LEFT / CENTER / RIGHT / UNKNOWN.

Pure logic: no OpenCV, MediaPipe or Windows dependencies, so it is fully
unit-testable without a camera or a display.

Threshold semantics
-------------------
The classifier works on one scalar ``x`` (the combined head+eye horizontal
signal, baseline-relative):

* ``|x| <= center_threshold``  -> CENTER
* ``|x| >= side_threshold``    -> LEFT / RIGHT (sign decides side)
* in between                   -> UNKNOWN (a deliberate ambiguous band)

The band between the two thresholds is the *spatial hysteresis*: it prevents
oscillation between CENTER and a side monitor for mid-range signals. Sign
convention: positive x = user looking toward their right = RIGHT monitor
(with the mirrored-frame convention; ``invert_x`` flips it if a particular
camera/setup turns out inverted).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .gaze_features import GazeBaseline, GazeSample


class MonitorState(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


CONCRETE_STATES = (MonitorState.LEFT, MonitorState.CENTER, MonitorState.RIGHT)


@dataclass
class Classification:
    state: MonitorState
    confidence: float
    x: float
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.state is not MonitorState.UNKNOWN


@dataclass
class ClassifierConfig:
    center_threshold: float = 0.12
    side_threshold: float = 0.32
    invert_x: bool = False
    auto_thresholds_from_noise: bool = False
    # Used only when auto_thresholds_from_noise is true:
    center_floor: float = 0.06
    side_floor: float = 0.16
    sigma_center_factor: float = 3.0
    sigma_side_factor: float = 7.0
    # Learned (three-monitor) calibration: x is -1 / 0 / +1 at the measured
    # monitors, boundaries sit half-way; this is the ambiguous band either
    # side of each boundary.
    monitor_band: float = 0.1


class MonitorClassifier:
    """Threshold classifier around the calibrated center baseline.

    Limitations (by design, MVP): the side thresholds are symmetric even
    though real side monitors rarely sit at symmetric angles. Tune
    ``center_threshold`` / ``side_threshold`` per setup (see README
    troubleshooting), or enable ``auto_thresholds_from_noise`` to derive them
    from calibration noise. Optional left/right anchor calibration can later
    be added without changing this interface.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        self.cfg = config or ClassifierConfig()
        self._baseline: Optional[GazeBaseline] = None

    @property
    def baseline(self) -> Optional[GazeBaseline]:
        return self._baseline

    def set_baseline(self, baseline: Optional[GazeBaseline]) -> None:
        self._baseline = baseline

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def learned(self) -> bool:
        return self._baseline is not None and self._baseline.model is not None

    def _thresholds(self) -> tuple[float, float]:
        cfg = self.cfg
        if self.learned:
            band = min(0.4, max(0.0, cfg.monitor_band))
            return 0.5 - band, 0.5 + band
        if cfg.auto_thresholds_from_noise and self._baseline is not None:
            sigma = max(1e-6, abs(self._baseline.std_x))
            center = max(cfg.center_floor, cfg.sigma_center_factor * sigma)
            side = max(cfg.side_floor, cfg.sigma_side_factor * sigma)
        else:
            center = cfg.center_threshold
            side = cfg.side_threshold
        # Guarantee an ambiguous band exists.
        side = max(side, center * 1.25, center + 0.02)
        return center, side

    def classify(self, sample: GazeSample) -> Classification:
        if self._baseline is None:
            return Classification(
                MonitorState.UNKNOWN, 0.0, sample.x, reason="not calibrated"
            )
        if not sample.valid:
            reason = "blink" if sample.blink >= 0.5 else "invalid features"
            return Classification(MonitorState.UNKNOWN, 0.0, sample.x, reason=reason)

        # A learned model already knows the direction; invert_x is for fixed weights.
        x = -sample.x if self.cfg.invert_x and not self.learned else sample.x
        center_thr, side_thr = self._thresholds()

        if abs(x) <= center_thr:
            conf = 0.5 + 0.5 * (1.0 - abs(x) / max(1e-6, center_thr))
            return Classification(MonitorState.CENTER, conf, x)
        if abs(x) >= side_thr:
            state = MonitorState.RIGHT if x > 0 else MonitorState.LEFT
            over = (abs(x) - center_thr) / max(1e-6, side_thr - center_thr)
            conf = 0.5 + 0.5 * min(1.0, over)
            return Classification(state, conf, x)
        return Classification(
            MonitorState.UNKNOWN, 0.25, x, reason="ambiguous (hysteresis band)"
        )
