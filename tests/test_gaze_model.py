"""Three-monitor calibration: the learned model must separate head rotation
from where the eyes point. Synthetic user, no camera required.

Ground truth: gaze angle theta (deg) = target * 30; while looking at a
monitor the head turns part of the way (phi ~ 0.6 * theta) and wiggles;
yaw feature = phi / 50, iris feature = (theta - phi) / 300.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.calibration import MonitorCalibration, fit_gaze_model, load_baseline, save_baseline
from src.face_tracker import FaceObservation
from src.gaze_features import FeatureExtractorConfig, GazeFeatureExtractor, GazeSample
from src.monitor_classifier import ClassifierConfig, MonitorClassifier, MonitorState

TARGETS = {"LEFT": -1.0, "CENTER": 0.0, "RIGHT": 1.0}


def features(theta: float, phi: float) -> tuple[float, float]:
    return phi / 50.0, (theta - phi) / 300.0


def synthetic(slot: str, n: int, rng: np.random.Generator) -> np.ndarray:
    theta = TARGETS[slot] * 30.0
    rows = []
    for _ in range(n):
        phi = 0.6 * theta + rng.uniform(-8, 8)
        yaw, iris = features(theta + rng.normal(0, 1.0), phi)
        rows.append((yaw + rng.normal(0, 0.005), iris + rng.normal(0, 0.002)))
    return np.array(rows)


def sample(yaw: float, iris: float, t_ms: int = 0) -> GazeSample:
    return GazeSample(
        t_ms=t_ms, valid=True, x=0.0, yaw_rel=yaw, pitch_rel=0.0, iris_dx=iris,
        iris_dy=0.0, blink=0.0, has_baseline=False,
        detail={"yaw_abs": yaw, "pitch_abs": 0.0, "iris_x_abs": iris, "iris_y_abs": 0.0},
    )


def run_calibration(seed: int = 1) -> MonitorCalibration:
    rng = np.random.default_rng(seed)
    calib = MonitorCalibration(step_s=2.0, settle_s=1.0, min_samples=30, max_step_s=6.0)
    t = 0
    calib.start(t)
    events = []
    while calib.running:
        slot = calib.step.value
        data = synthetic(slot, 1, rng)[0]
        calib.add_sample(sample(data[0], data[1], t), t)
        event = calib.update(t)
        if event:
            events.append(event)
        if event == "done":
            break
        t += 33
    assert events == ["collect", "step", "collect", "step", "collect", "done"]
    return calib


def test_fit_learns_eye_compensation():
    rng = np.random.default_rng(0)
    model = fit_gaze_model({s: synthetic(s, 120, rng) for s in TARGETS})
    assert model is not None
    # theta = 50*yaw + 300*iris, so the iris weight is ~6x the yaw weight
    assert model.w_iris / model.w_yaw == pytest.approx(6.0, rel=0.25)
    assert model.anchors["LEFT"] < model.anchors["CENTER"] < model.anchors["RIGHT"]


def test_head_turned_left_eyes_on_center_is_center():
    """The reported bug: head turned well left, eyes on the left half of the
    center monitor. Fixed weights said LEFT; the learned model says CENTER."""
    result = run_calibration().finish()
    assert result.ok and result.baseline is not None and result.baseline.model is not None

    yaw, iris = features(theta=-8.0, phi=-20.0)
    obs = FaceObservation(t_ms=0, yaw=yaw, iris_dx=iris)
    extractor = GazeFeatureExtractor(FeatureExtractorConfig())
    classifier = MonitorClassifier(ClassifierConfig())

    classifier.set_baseline(result.baseline)
    learned = classifier.classify(extractor.extract(obs, result.baseline, 0))
    assert learned.state is MonitorState.CENTER

    old = result.baseline
    old.model = None  # same user, fixed-weight formula
    classifier.set_baseline(old)
    fixed = classifier.classify(extractor.extract(obs, old, 0))
    assert fixed.state is MonitorState.LEFT


@pytest.mark.parametrize("theta,phi,expected", [
    (-30.0, -18.0, MonitorState.LEFT),
    (-30.0, -5.0, MonitorState.LEFT),  # eyes do the work, head barely turned
    (30.0, 25.0, MonitorState.RIGHT),
    (0.0, 10.0, MonitorState.CENTER),  # head turned right, eyes back on center
])
def test_learned_model_classifies_by_gaze(theta, phi, expected):
    result = run_calibration(seed=3).finish()
    classifier = MonitorClassifier(ClassifierConfig())
    classifier.set_baseline(result.baseline)
    yaw, iris = features(theta, phi)
    obs = FaceObservation(t_ms=0, yaw=yaw, iris_dx=iris)
    gaze = GazeFeatureExtractor(FeatureExtractorConfig()).extract(obs, result.baseline, 0)
    assert classifier.classify(gaze).state is expected


def test_head_circle_fills_coverage_and_ends_step_early():
    calib = MonitorCalibration(step_s=8.0, settle_s=1.0, min_samples=20, max_step_s=14.0)
    calib.start(0)
    t, events = 0, []
    while calib.step_index == 0 and calib.running:
        a = t / 1000.0 * 2.0  # one head circle every ~3 s
        moving = calib.collecting(t) and calib.n_valid >= 6
        yaw, pitch = (0.12 * math.cos(a), 0.08 * math.sin(a)) if moving else (0.0, 0.0)
        s = sample(yaw, 0.0, t)
        s.detail["pitch_abs"] = pitch
        calib.add_sample(s, t)
        if (event := calib.update(t)):
            events.append((event, t))
        if calib.step_index == 0:
            t += 33
    assert events[-1][0] == "step"
    collected_s = events[-1][1] / 1000.0 - 1.0
    assert 4.8 <= collected_s < 8.0  # full circle: done before the 8 s limit
    assert calib.coverage == tuple([False] * len(calib.coverage))  # reset for the next monitor


def test_coverage_directions_match_preview():
    calib = MonitorCalibration(settle_s=0.0, min_samples=5)
    calib.start(0)
    for i in range(6):
        calib.add_sample(sample(0.0, 0.0, i), i)
    calib.add_sample(sample(0.2, 0.0, 10), 10)  # head to the user's right
    cov = calib.coverage
    assert cov[0] and not cov[len(cov) // 2]  # bin 0 = right side of the ring


def test_missing_monitor_fails_with_its_name():
    calib = MonitorCalibration(step_s=1.0, settle_s=0.5, min_samples=10, max_step_s=2.0)
    calib.start(0)
    t = 0
    while calib.running:
        if calib.step is not MonitorState.LEFT:  # face lost while looking left
            calib.add_sample(sample(0.0, 0.0, t), t)
        if calib.update(t) == "done":
            break
        t += 50
    result = calib.finish()
    assert not result.ok
    assert "left" in result.message


def test_indistinguishable_monitors_fail():
    calib = MonitorCalibration(step_s=1.0, settle_s=0.5, min_samples=10)
    calib.start(0)
    t = 0
    rng = np.random.default_rng(5)
    while calib.running:
        calib.add_sample(sample(rng.normal(0, 0.01), rng.normal(0, 0.002), t), t)
        if calib.update(t) == "done":
            break
        t += 33
    assert not calib.finish().ok


def test_model_survives_save_and_load(tmp_path):
    result = run_calibration().finish()
    path = tmp_path / "baseline.json"
    save_baseline(path, result.baseline)
    loaded = load_baseline(path)
    assert loaded.model is not None
    assert loaded.model.anchors == pytest.approx(result.baseline.model.anchors)
    assert loaded.model.w_iris == pytest.approx(result.baseline.model.w_iris)
