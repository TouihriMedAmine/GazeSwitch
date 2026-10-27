"""Classifier tests: LEFT/CENTER/RIGHT/UNKNOWN decisions, thresholds,
hysteresis band, sign conventions, confidence bounds."""
from __future__ import annotations

import pytest

from src.gaze_features import GazeBaseline
from src.monitor_classifier import ClassifierConfig, MonitorClassifier, MonitorState


@pytest.fixture
def baseline():
    return GazeBaseline(
        yaw=0.0,
        pitch=0.0,
        iris_x=0.0,
        iris_y=0.0,
        std_yaw=0.01,
        std_iris_x=0.01,
        std_x=0.01,
        n_samples=50,
        created_at="2026-01-01T00:00:00",
    )


def make_sample(x: float, valid: bool = True, blink: float = 0.0) -> "GazeSample":
    from src.gaze_features import GazeSample

    return GazeSample(t_ms=0, valid=valid, x=x, blink=blink, has_baseline=True)


CFG = ClassifierConfig(center_threshold=0.12, side_threshold=0.32)


def test_not_calibrated_is_unknown():
    clf = MonitorClassifier(CFG)
    result = clf.classify(make_sample(0.0))
    assert result.state is MonitorState.UNKNOWN
    assert result.reason == "not calibrated"
    assert not clf.calibrated


def test_center_state(baseline):
    clf = MonitorClassifier(CFG)
    clf.set_baseline(baseline)
    assert clf.classify(make_sample(0.0)).state is MonitorState.CENTER
    assert clf.classify(make_sample(0.10)).state is MonitorState.CENTER
    assert clf.classify(make_sample(-0.12)).state is MonitorState.CENTER


def test_side_states(baseline):
    clf = MonitorClassifier(CFG)
    clf.set_baseline(baseline)
    assert clf.classify(make_sample(0.32)).state is MonitorState.RIGHT
    assert clf.classify(make_sample(0.9)).state is MonitorState.RIGHT
    assert clf.classify(make_sample(-0.32)).state is MonitorState.LEFT
    assert clf.classify(make_sample(-0.9)).state is MonitorState.LEFT


def test_hysteresis_band_is_unknown(baseline):
    """Values strictly between the thresholds must be UNKNOWN - this band is
    the spatial hysteresis that prevents center/side oscillation."""
    clf = MonitorClassifier(CFG)
    clf.set_baseline(baseline)
    for x in (0.13, 0.2, 0.31, -0.2, -0.25):
        result = clf.classify(make_sample(x))
        assert result.state is MonitorState.UNKNOWN, f"x={x} should be ambiguous"
        assert result.reason.startswith("ambiguous")


def test_blink_and_invalid_are_unknown(baseline):
    clf = MonitorClassifier(CFG)
    clf.set_baseline(baseline)
    assert clf.classify(make_sample(0.9, valid=False)).state is MonitorState.UNKNOWN
    blinked = clf.classify(make_sample(0.9, valid=False, blink=0.9))
    assert blinked.state is MonitorState.UNKNOWN
    assert blinked.reason == "blink"


def test_invert_x_flips_sides(baseline):
    clf = MonitorClassifier(ClassifierConfig(center_threshold=0.12, side_threshold=0.32,
                                             invert_x=True))
    clf.set_baseline(baseline)
    assert clf.classify(make_sample(0.5)).state is MonitorState.LEFT
    assert clf.classify(make_sample(-0.5)).state is MonitorState.RIGHT


def test_confidence_within_bounds(baseline):
    clf = MonitorClassifier(CFG)
    clf.set_baseline(baseline)
    for x in (-1.0, -0.4, -0.05, 0.0, 0.05, 0.4, 1.0):
        result = clf.classify(make_sample(x))
        if result.state is not MonitorState.UNKNOWN:
            assert 0.0 <= result.confidence <= 1.0


def test_auto_thresholds_from_noise(baseline):
    baseline.std_x = 0.05  # noisy calibration
    clf = MonitorClassifier(ClassifierConfig(auto_thresholds_from_noise=True))
    clf.set_baseline(baseline)
    center_thr, side_thr = clf._thresholds()
    # 3*sigma and 7*sigma with floors
    assert center_thr == pytest.approx(max(0.06, 3 * 0.05))
    assert side_thr == pytest.approx(max(0.16, 7 * 0.05))
    assert side_thr > center_thr
    # Fixed thresholds must be used when the flag is off
    clf2 = MonitorClassifier(CFG)
    clf2.set_baseline(baseline)
    assert clf2._thresholds() == (0.12, 0.32)


def test_side_threshold_always_exceeds_center(baseline):
    clf = MonitorClassifier(ClassifierConfig(center_threshold=0.5, side_threshold=0.4))
    clf.set_baseline(baseline)
    center_thr, side_thr = clf._thresholds()
    assert side_thr > center_thr
