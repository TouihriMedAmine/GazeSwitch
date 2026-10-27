"""Gaze feature extraction and the calibration baseline model.

This module is pure logic (numpy only): it turns a :class:`FaceObservation`
into a normalised, baseline-relative :class:`GazeSample` that the monitor
classifier consumes. It never touches OpenCV, MediaPipe or Win32 directly,
which keeps the estimation pipeline unit-testable.

Conventions
-----------
* Frames are processed in *mirrored* (selfie-view) form when ``camera.mirror``
  is enabled. In a mirrored frame, "positive x" means "toward the right edge
  of the image", which corresponds to the **user's right** (i.e. a monitor
  placed on the user's right-hand side).
* ``yaw`` is an *approximate* head-turn proxy in pseudo-radians; positive =
  user looking toward their right. It is NOT a calibrated physical angle.
* ``iris_dx`` is the average relative iris offset inside both eye openings,
  normalised by eye width; positive = iris toward the user's right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .face_tracker import FaceObservation

BASELINE_VERSION = 2


@dataclass
class GazeModel:
    """Per-user gaze model learned from the three-monitor calibration.

    ``raw = w_yaw * yaw + w_iris * iris + bias`` on absolute features, fitted
    so the left / center / right monitors land near -1 / 0 / +1 while the
    head moves - which teaches how much the eyes compensate for head turns.
    ``anchors`` are the measured raw values for each monitor; ``normalize``
    maps raw onto a piecewise-linear scale where the anchors sit exactly at
    -1 / 0 / +1 (handles monitors at uneven angles).
    """

    w_yaw: float
    w_iris: float
    bias: float
    anchors: Dict[str, float]

    def raw(self, yaw: float, iris: float) -> float:
        return self.w_yaw * yaw + self.w_iris * iris + self.bias

    def normalize(self, raw: float) -> float:
        left, center, right = self.anchors["LEFT"], self.anchors["CENTER"], self.anchors["RIGHT"]
        if raw < center:
            return (raw - center) / max(1e-6, center - left)
        return (raw - center) / max(1e-6, right - center)

    def to_dict(self) -> Dict[str, Any]:
        return {"w_yaw": self.w_yaw, "w_iris": self.w_iris, "bias": self.bias,
                "anchors": dict(self.anchors)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GazeModel":
        anchors = {k: float(data["anchors"][k]) for k in ("LEFT", "CENTER", "RIGHT")}
        return cls(float(data["w_yaw"]), float(data["w_iris"]), float(data["bias"]), anchors)


@dataclass
class GazeBaseline:
    """Neutral 'looking at the center monitor' feature vector."""

    yaw: float
    pitch: float
    iris_x: float
    iris_y: float
    std_yaw: float
    std_iris_x: float
    std_x: float  # noise estimate of the combined x signal during calibration
    n_samples: int
    created_at: str
    camera_index: Optional[int] = None
    version: int = BASELINE_VERSION
    model: Optional[GazeModel] = None  # None = center-only (fixed weights)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "version": self.version,
            "created_at": self.created_at,
            "camera_index": self.camera_index,
            "n_samples": self.n_samples,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "iris_x": self.iris_x,
            "iris_y": self.iris_y,
            "std_yaw": self.std_yaw,
            "std_iris_x": self.std_iris_x,
            "std_x": self.std_x,
        }
        if self.model is not None:
            data["model"] = self.model.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GazeBaseline":
        required = (
            "yaw",
            "pitch",
            "iris_x",
            "iris_y",
            "std_yaw",
            "std_iris_x",
            "std_x",
            "n_samples",
            "created_at",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"baseline file missing keys: {missing}")
        if int(data.get("version", 0)) > BASELINE_VERSION:
            raise ValueError(
                f"baseline version {data.get('version')} is newer than supported "
                f"{BASELINE_VERSION}; please recalibrate"
            )
        return cls(
            yaw=float(data["yaw"]),
            pitch=float(data["pitch"]),
            iris_x=float(data["iris_x"]),
            iris_y=float(data["iris_y"]),
            std_yaw=float(data["std_yaw"]),
            std_iris_x=float(data["std_iris_x"]),
            std_x=float(data["std_x"]),
            n_samples=int(data["n_samples"]),
            created_at=str(data["created_at"]),
            camera_index=(
                int(data["camera_index"]) if data.get("camera_index") is not None else None
            ),
            version=int(data.get("version", BASELINE_VERSION)),
            model=GazeModel.from_dict(data["model"]) if data.get("model") else None,
        )


@dataclass
class GazeSample:
    """Baseline-relative gaze features for a single frame."""

    t_ms: int
    valid: bool
    x: float = 0.0  # combined horizontal signal (roughly -1..+1)
    y: float = 0.0  # combined vertical signal (roughly -1..+1; for future use)
    yaw_rel: float = 0.0
    pitch_rel: float = 0.0
    iris_dx: float = 0.0
    iris_dy: float = 0.0
    blink: float = 0.0
    has_baseline: bool = False
    detail: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureExtractorConfig:
    weight_head: float = 0.65
    weight_iris: float = 0.35
    yaw_scale: float = 0.45  # pseudo-radians that map to a "full" head signal
    iris_scale: float = 0.18  # iris ratio change that maps to a "full" eye signal
    pitch_scale: float = 0.35
    iris_y_scale: float = 0.15


class GazeFeatureExtractor:
    """Combines head pose and relative iris position into one gaze signal.

    The combined signal is a transparent weighted sum; no classifier is
    involved. Head direction is deliberately *not* treated as gaze direction:
    it is one weighted contribution among two.
    """

    def __init__(self, config: Optional[FeatureExtractorConfig] = None) -> None:
        self.cfg = config or FeatureExtractorConfig()

    def extract(
        self,
        obs: Optional[FaceObservation],
        baseline: Optional[GazeBaseline],
        t_ms: int,
    ) -> GazeSample:
        if obs is None:
            return GazeSample(t_ms=t_ms, valid=False, has_baseline=baseline is not None)

        yaw_rel = obs.yaw - (baseline.yaw if baseline else 0.0)
        pitch_rel = obs.pitch - (baseline.pitch if baseline else 0.0)
        iris_dx = obs.iris_dx - (baseline.iris_x if baseline else 0.0)
        iris_dy = obs.iris_dy - (baseline.iris_y if baseline else 0.0)

        finite = all(
            math.isfinite(v) for v in (yaw_rel, pitch_rel, iris_dx, iris_dy, obs.blink)
        )
        blinking = obs.blink >= 0.5
        valid = finite and not blinking

        hn = yaw_rel / self.cfg.yaw_scale
        en = iris_dx / self.cfg.iris_scale
        wsum = self.cfg.weight_head + self.cfg.weight_iris
        if baseline is not None and baseline.model is not None:
            x = baseline.model.normalize(baseline.model.raw(obs.yaw, obs.iris_dx))
        else:
            x = (self.cfg.weight_head * hn + self.cfg.weight_iris * en) / wsum

        pv = pitch_rel / self.cfg.pitch_scale
        ey = iris_dy / self.cfg.iris_y_scale
        y = (self.cfg.weight_head * pv + self.cfg.weight_iris * ey) / wsum

        return GazeSample(
            t_ms=t_ms,
            valid=valid,
            x=x,
            y=y,
            yaw_rel=yaw_rel,
            pitch_rel=pitch_rel,
            iris_dx=iris_dx,
            iris_dy=iris_dy,
            blink=obs.blink,
            has_baseline=baseline is not None,
            detail={
                "yaw_abs": obs.yaw,
                "pitch_abs": obs.pitch,
                "iris_x_abs": obs.iris_dx,
                "iris_y_abs": obs.iris_dy,
            },
        )
