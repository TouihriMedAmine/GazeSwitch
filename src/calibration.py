"""Center-gaze calibration: collect frames, estimate a stable baseline.

The calibration collects baseline-relative feature *inputs* (raw absolute
values) while the user looks at the center monitor, then stores the median
and a robust noise estimate (1.4826 * MAD) per feature. The baseline can be
saved to / loaded from a small JSON file so it can be reused between
sessions.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gaze_features import GazeBaseline, GazeModel, GazeSample
from .monitor_classifier import MonitorState

log = logging.getLogger(__name__)

BASELINE_FORMAT_VERSION = 1


class CalibrationError(RuntimeError):
    pass


@dataclass
class CalibrationResult:
    ok: bool
    message: str
    baseline: Optional[GazeBaseline]
    n_valid: int
    n_invalid: int
    noisy: bool = False


class CenterCalibration:
    def __init__(
        self,
        duration_s: float = 3.0,
        min_samples: int = 40,
        max_noise_x: float = 0.12,
        max_duration_s: float = 10.0,
    ) -> None:
        self.duration_s = duration_s
        self.min_samples = int(min_samples)
        # Low-FPS cameras cannot deliver min_samples within duration_s, so
        # collection is extended up to this hard limit.
        self.max_duration_s = max(duration_s, max_duration_s)
        self.max_noise_x = max_noise_x
        self._yaw: List[float] = []
        self._pitch: List[float] = []
        self._iris_x: List[float] = []
        self._iris_y: List[float] = []
        self._combined_x: List[float] = []
        self._n_invalid = 0
        self._started_ms: Optional[int] = None

    # ------------------------------------------------------------------ API
    def start(self, now_ms: int) -> None:
        self._yaw.clear()
        self._pitch.clear()
        self._iris_x.clear()
        self._iris_y.clear()
        self._combined_x.clear()
        self._n_invalid = 0
        self._started_ms = now_ms
        log.info("Calibration started (%.1fs, need >= %d valid samples)",
                 self.duration_s, self.min_samples)

    @property
    def running(self) -> bool:
        return self._started_ms is not None

    @property
    def n_valid(self) -> int:
        return len(self._yaw)

    @property
    def n_invalid(self) -> int:
        return self._n_invalid

    def progress(self, now_ms: int) -> float:
        if self._started_ms is None:
            return 0.0
        elapsed = (now_ms - self._started_ms) / 1000.0
        return max(0.0, min(1.0, elapsed / max(1e-6, self.duration_s)))

    def add_sample(self, sample: GazeSample) -> None:
        """Record one frame. Invalid frames (no face, blink, NaN) are counted
        and rejected."""
        if not self.running:
            return
        ok = sample.valid and all(
            math.isfinite(v)
            for v in (
                sample.detail.get("yaw_abs", 0.0),
                sample.detail.get("pitch_abs", 0.0),
                sample.detail.get("iris_x_abs", 0.0),
                sample.detail.get("iris_y_abs", 0.0),
            )
        )
        if not ok:
            self._n_invalid += 1
            return
        self._yaw.append(sample.detail["yaw_abs"])
        self._pitch.append(sample.detail["pitch_abs"])
        self._iris_x.append(sample.detail["iris_x_abs"])
        self._iris_y.append(sample.detail["iris_y_abs"])
        self._combined_x.append(sample.x)

    def finished(self, now_ms: int) -> bool:
        if self._started_ms is None:
            return False
        elapsed = (now_ms - self._started_ms) / 1000.0
        if elapsed >= self.max_duration_s:
            return True
        return elapsed >= self.duration_s and self.n_valid >= self.min_samples

    def finish(self) -> CalibrationResult:
        """Compute the baseline from the collected samples and stop."""
        self._started_ms = None
        n = self.n_valid
        if n < self.min_samples:
            log.warning(
                "Calibration FAILED: %d valid samples, need >= %d (%d rejected). "
                "Raise calibration.max_duration_s if the camera FPS is low.",
                n, self.min_samples, self._n_invalid,
            )
            return CalibrationResult(
                ok=False,
                message="Calibration didn't finish. Face the camera, hold still, "
                        "and press C to try again.",
                baseline=None,
                n_valid=n,
                n_invalid=self._n_invalid,
            )

        yaw = float(statistics.median(self._yaw))
        pitch = float(statistics.median(self._pitch))
        iris_x = float(statistics.median(self._iris_x))
        iris_y = float(statistics.median(self._iris_y))

        def robust_std(values: List[float]) -> float:
            med = statistics.median(values)
            mad = statistics.median([abs(v - med) for v in values])
            return max(1e-6, 1.4826 * mad)

        std_x = float(statistics.pstdev(self._combined_x)) if len(self._combined_x) > 1 else 0.02
        noisy = std_x > self.max_noise_x
        baseline = GazeBaseline(
            yaw=yaw,
            pitch=pitch,
            iris_x=iris_x,
            iris_y=iris_y,
            std_yaw=robust_std(self._yaw),
            std_iris_x=robust_std(self._iris_x),
            std_x=max(1e-4, std_x),
            n_samples=n,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        log.info("Calibration OK: %d samples (rejected %d). Noise x=%.3f%s",
                 n, self._n_invalid, std_x, " [HIGH]" if noisy else "")
        message = (
            "Calibrated. You moved a little - recalibrate for better accuracy."
            if noisy else "Calibrated"
        )
        return CalibrationResult(
            ok=True,
            message=message,
            baseline=baseline,
            n_valid=n,
            n_invalid=self._n_invalid,
            noisy=noisy,
        )


# --------------------------------------------------------- three monitors
CALIBRATION_STEPS = (MonitorState.CENTER, MonitorState.LEFT, MonitorState.RIGHT)
_TARGETS = {"LEFT": -1.0, "CENTER": 0.0, "RIGHT": 1.0}


def _valid_features(sample: GazeSample) -> Optional[Tuple[float, float, float, float]]:
    if not sample.valid:
        return None
    values = tuple(sample.detail.get(k, float("nan"))
                   for k in ("yaw_abs", "pitch_abs", "iris_x_abs", "iris_y_abs"))
    return values if all(math.isfinite(v) for v in values) else None  # type: ignore[return-value]


def fit_gaze_model(
    samples: Dict[str, np.ndarray],
    prior: Tuple[float, float] = (1.44, 1.94),
    ridge: float = 0.02,
) -> Optional[GazeModel]:
    """Fit raw = w_yaw*yaw + w_iris*iris + bias so each monitor's samples land
    on its target (-1 / 0 / +1).

    ``samples[slot]`` is an (N, 2) array of absolute (yaw, iris) features
    recorded while the user looked at that monitor and turned their head:
    within one monitor the gaze is fixed while the head moves, which is what
    separates the eye contribution from the head contribution. Ridge pulls
    toward ``prior`` (the fixed default weights) when the data can't decide.
    Returns None when the monitors can't be told apart.
    """
    feats = np.concatenate([samples[s] for s in _TARGETS])
    target = np.concatenate([np.full(len(samples[s]), _TARGETS[s]) for s in _TARGETS])
    mean, std = feats.mean(axis=0), np.maximum(feats.std(axis=0), [1e-3, 1e-4])
    z = (feats - mean) / std
    lam = ridge * len(z)

    def solve(columns: List[int]) -> np.ndarray:
        design = np.column_stack([z[:, columns], np.ones(len(z))])
        penalty = np.diag([lam] * len(columns) + [0.0])
        pull = np.array([prior[c] * std[c] * lam for c in columns] + [0.0])
        return np.linalg.solve(design.T @ design + penalty, design.T @ target + pull)
    # The iris weight's sign is left free: in practice the eye-corner ratio can
    # move with or against the head depending on camera angle.

    coef = solve([0, 1])
    w_yaw, w_iris = coef[0] / std[0], coef[1] / std[1]
    if w_yaw <= 0:
        return None
    bias = coef[-1] - w_yaw * mean[0] - w_iris * mean[1]

    anchors = {
        s: float(np.median(w_yaw * samples[s][:, 0] + w_iris * samples[s][:, 1] + bias))
        for s in _TARGETS
    }
    if not anchors["LEFT"] < anchors["CENTER"] < anchors["RIGHT"]:
        return None
    return GazeModel(float(w_yaw), float(w_iris), float(bias), anchors)


COVERAGE_BINS = 36


class MonitorCalibration:
    """Guided calibration: look at the center, left, then right monitor while
    moving the head around. Each step has a settle phase (time to move the
    eyes there; samples ignored) followed by a collect phase.

    During collection the head direction relative to the step's starting
    pose fills ``coverage`` (one flag per direction, Face ID style); a step
    ends early once the circle is nearly complete and enough time passed."""

    def __init__(
        self,
        step_s: float = 8.0,
        settle_s: float = 3.0,
        min_samples: int = 40,
        max_step_s: float = 14.0,
        max_noise_x: float = 0.12,
        prior: Tuple[float, float] = (1.44, 1.94),
        monitor_band: float = 0.1,
        yaw_reach: float = 0.08,
        pitch_reach: float = 0.05,
        complete_coverage: float = 0.9,
        min_step_fraction: float = 0.6,
    ) -> None:
        self.yaw_reach = yaw_reach
        self.pitch_reach = pitch_reach
        self.complete_coverage = complete_coverage
        self.min_step_fraction = min_step_fraction
        self._coverage = [False] * COVERAGE_BINS
        self._ref: Optional[Tuple[float, float]] = None
        self.steps = CALIBRATION_STEPS
        self.step_s = step_s
        self.settle_s = settle_s
        self.min_samples = int(min_samples)
        self.max_step_s = max(step_s, max_step_s)
        self.max_noise_x = max_noise_x
        self.prior = prior
        self.monitor_band = monitor_band
        self._samples: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self._index = 0
        self._step_started_ms = 0
        self._collect_announced = False
        self._running = False
        self._n_invalid = 0

    # ------------------------------------------------------------------ API
    def start(self, now_ms: int) -> None:
        self._samples = {s.value: [] for s in self.steps}
        self._index = 0
        self._begin_step(now_ms)
        self._n_invalid = 0
        self._running = True
        log.info("Calibration started (%d monitors, %.1fs each)", len(self.steps), self.step_s)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def step(self) -> Optional[MonitorState]:
        return self.steps[self._index] if self._running else None

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def n_valid(self) -> int:
        step = self.step
        return len(self._samples[step.value]) if step else 0

    @property
    def n_invalid(self) -> int:
        return self._n_invalid

    @property
    def coverage(self) -> Tuple[bool, ...]:
        """Head directions reached this step; bin 0 = user's right, then
        clockwise in the mirrored preview (down, left, up)."""
        return tuple(self._coverage)

    @property
    def coverage_fraction(self) -> float:
        return sum(self._coverage) / COVERAGE_BINS

    def collecting(self, now_ms: int) -> bool:
        return self._running and now_ms - self._step_started_ms >= self.settle_s * 1000

    def just_completed_step(self, now_ms: int, window_ms: int = 700) -> bool:
        """True right after moving on from a finished monitor (for a green flash)."""
        return self._running and self._index > 0 and now_ms - self._step_started_ms < window_ms

    def settle_remaining(self, now_ms: int) -> float:
        """Seconds until the current step starts collecting."""
        return max(0.0, self.settle_s - (now_ms - self._step_started_ms) / 1000.0)

    def progress(self, now_ms: int) -> float:
        """Progress of the current step's collect phase, 0..1."""
        if not self.collecting(now_ms):
            return 0.0
        elapsed = (now_ms - self._step_started_ms) / 1000.0 - self.settle_s
        return max(0.0, min(1.0, elapsed / max(1e-6, self.step_s)))

    def add_sample(self, sample: GazeSample, now_ms: int) -> None:
        if not self.collecting(now_ms):
            return
        features = _valid_features(sample)
        if features is None:
            self._n_invalid += 1
            return
        samples = self._samples[self.steps[self._index].value]
        samples.append(features)
        self._track_direction(samples, features[0], features[1])

    def _begin_step(self, now_ms: int) -> None:
        self._step_started_ms = now_ms
        self._collect_announced = False
        self._coverage = [False] * COVERAGE_BINS
        self._ref = None

    def _track_direction(self, samples, yaw: float, pitch: float) -> None:
        if self._ref is None:
            if len(samples) < 6:
                return
            first = np.asarray(samples[:6])
            self._ref = (float(np.median(first[:, 0])), float(np.median(first[:, 1])))
        dx = (yaw - self._ref[0]) / self.yaw_reach
        dy = (pitch - self._ref[1]) / self.pitch_reach
        if math.hypot(dx, dy) < 1.0:
            return
        angle = math.atan2(dy, dx) % (2 * math.pi)
        center = int(angle / (2 * math.pi) * COVERAGE_BINS) % COVERAGE_BINS
        for k in (-1, 0, 1):
            self._coverage[(center + k) % COVERAGE_BINS] = True

    def update(self, now_ms: int) -> Optional[str]:
        """Advance the step machine. Returns "collect" when a collect phase
        starts, "step" when moving to the next monitor, "done" at the end."""
        if not self._running:
            return None
        elapsed = (now_ms - self._step_started_ms) / 1000.0
        if not self._collect_announced and elapsed >= self.settle_s:
            self._collect_announced = True
            return "collect"
        collected = elapsed - self.settle_s
        enough = self.n_valid >= self.min_samples
        circle_done = (self.coverage_fraction >= self.complete_coverage
                       and collected >= self.step_s * self.min_step_fraction)
        if ((collected >= self.step_s or circle_done) and enough) or collected >= self.max_step_s:
            log.info("Calibration step %s: %d samples, head directions %.0f%%",
                     self.steps[self._index].value, self.n_valid, self.coverage_fraction * 100)
            self._index += 1
            self._begin_step(now_ms)
            if self._index >= len(self.steps):
                self._index = len(self.steps) - 1
                return "done"
            return "step"
        return None

    def finish(self) -> CalibrationResult:
        self._running = False
        counts = {slot: len(v) for slot, v in self._samples.items()}
        n_total = sum(counts.values())
        short = [slot for slot, n in counts.items() if n < self.min_samples]
        if short:
            log.warning("Calibration FAILED: samples per monitor %s, need >= %d (%d rejected)",
                        counts, self.min_samples, self._n_invalid)
            where = " and ".join(s.lower() for s in short)
            return CalibrationResult(
                ok=False,
                message=f"Calibration didn't finish: couldn't see your face while you "
                        f"looked at the {where} monitor. Face the camera and press C to "
                        f"try again.",
                baseline=None, n_valid=n_total, n_invalid=self._n_invalid,
            )

        arrays = {slot: np.asarray(v, dtype=np.float64) for slot, v in self._samples.items()}
        model = fit_gaze_model({s: a[:, [0, 2]] for s, a in arrays.items()}, prior=self.prior)
        if model is None:
            log.warning("Calibration FAILED: monitors not separable (samples %s)", counts)
            return CalibrationResult(
                ok=False,
                message="Calibration couldn't tell your monitors apart. Turn toward "
                        "each monitor when asked, and press C to try again.",
                baseline=None, n_valid=n_total, n_invalid=self._n_invalid,
            )

        center = arrays["CENTER"]
        x_by_slot = {s: np.array([model.normalize(model.raw(r[0], r[2])) for r in a])
                     for s, a in arrays.items()}
        lo, hi = 0.5 - self.monitor_band, 0.5 + self.monitor_band
        hits = (np.sum(np.abs(x_by_slot["CENTER"]) <= lo)
                + np.sum(x_by_slot["LEFT"] <= -hi) + np.sum(x_by_slot["RIGHT"] >= hi))
        accuracy = hits / n_total
        std_x = float(np.std(x_by_slot["CENTER"]))

        def robust_std(values: np.ndarray) -> float:
            med = np.median(values)
            return max(1e-6, 1.4826 * float(np.median(np.abs(values - med))))

        baseline = GazeBaseline(
            yaw=float(np.median(center[:, 0])),
            pitch=float(np.median(center[:, 1])),
            iris_x=float(np.median(center[:, 2])),
            iris_y=float(np.median(center[:, 3])),
            std_yaw=robust_std(center[:, 0]),
            std_iris_x=robust_std(center[:, 2]),
            std_x=max(1e-4, std_x),
            n_samples=n_total,
            created_at=datetime.now().isoformat(timespec="seconds"),
            model=model,
        )
        log.info(
            "Calibration OK: samples %s (rejected %d), w_yaw=%.3f w_iris=%.3f bias=%.3f "
            "anchors L=%.3f C=%.3f R=%.3f, calibration accuracy %.0f%%",
            counts, self._n_invalid, model.w_yaw, model.w_iris, model.bias,
            model.anchors["LEFT"], model.anchors["CENTER"], model.anchors["RIGHT"],
            accuracy * 100,
        )
        noisy = accuracy < 0.8
        message = ("Calibrated, but some looks were unclear. Recalibrate for better "
                   "accuracy." if noisy else "Calibrated for all three monitors")
        return CalibrationResult(ok=True, message=message, baseline=baseline,
                                 n_valid=n_total, n_invalid=self._n_invalid, noisy=noisy)


# ------------------------------------------------------------------ storage
def save_baseline(path: str | Path, baseline: GazeBaseline) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
    log.info("Baseline saved to %s", p)


def load_baseline(path: str | Path) -> GazeBaseline:
    p = Path(path)
    if not p.exists():
        raise CalibrationError(f"Baseline file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        baseline = GazeBaseline.from_dict(data)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        raise CalibrationError(f"Baseline file {p} is invalid: {exc}") from exc
    if baseline.n_samples < 10:
        raise CalibrationError(
            f"Baseline file {p} has only {baseline.n_samples} samples; recalibrate."
        )
    log.info("Baseline loaded from %s (n=%d, created %s)", p, baseline.n_samples,
             baseline.created_at)
    return baseline
