"""Calibration tests: robust baseline estimation, rejection of invalid
frames, save/load round-trip. No camera required."""
from __future__ import annotations

import math

import pytest

from src.calibration import CalibrationError, CenterCalibration, load_baseline, save_baseline
from src.gaze_features import GazeSample


def make_sample(valid=True, yaw=0.0, pitch=0.0, iris_x=0.0, iris_y=0.0, blink=0.0, x=0.0,
                t_ms=0) -> GazeSample:
    return GazeSample(
        t_ms=t_ms,
        valid=valid,
        x=x,
        yaw_rel=yaw,
        pitch_rel=pitch,
        iris_dx=iris_x,
        iris_dy=iris_y,
        blink=blink,
        has_baseline=False,
        detail={"yaw_abs": yaw, "pitch_abs": pitch, "iris_x_abs": iris_x, "iris_y_abs": iris_y},
    )


def test_insufficient_samples_fails():
    calib = CenterCalibration(duration_s=3.0, min_samples=40)
    calib.start(now_ms=0)
    for i in range(10):  # far below min_samples
        calib.add_sample(make_sample(yaw=0.1, x=0.05, t_ms=i))
    result = calib.finish()
    assert not result.ok
    assert result.baseline is None
    assert result.n_valid == 10


def test_baseline_median_is_robust_to_outliers():
    calib = CenterCalibration(duration_s=3.0, min_samples=10)
    calib.start(now_ms=0)
    for i in range(50):
        # stable yaw ~0.10 with one huge outlier
        yaw = 0.10 if i != 25 else 5.0
        calib.add_sample(make_sample(yaw=yaw, pitch=0.02, iris_x=0.01, x=0.02, t_ms=i))
    result = calib.finish()
    assert result.ok
    assert result.baseline is not None
    assert math.isclose(result.baseline.yaw, 0.10, abs_tol=1e-6)
    assert math.isclose(result.baseline.pitch, 0.02, abs_tol=1e-6)
    assert math.isclose(result.baseline.iris_x, 0.01, abs_tol=1e-6)


def test_invalid_and_blinking_frames_are_rejected():
    calib = CenterCalibration(duration_s=3.0, min_samples=10)
    calib.start(now_ms=0)
    calib.add_sample(make_sample(valid=False, yaw=1.0))   # invalid -> rejected
    calib.add_sample(make_sample(blink=0.9, yaw=1.0))     # blink -> rejected (extractor
    # marks samples with blink>=0.5 invalid; simulate by valid=False too)
    calib.add_sample(make_sample(yaw=float("nan")))       # NaN -> rejected
    for i in range(12):
        calib.add_sample(make_sample(yaw=0.05, t_ms=i))
    result = calib.finish()
    assert result.ok
    assert result.n_invalid >= 2
    assert math.isclose(result.baseline.yaw, 0.05, abs_tol=1e-6)


def test_progress_and_duration():
    calib = CenterCalibration(duration_s=2.0, min_samples=5)
    calib.start(now_ms=0)
    assert calib.progress(0) == 0.0
    assert calib.progress(1000) == pytest.approx(0.5)
    assert calib.progress(2000) == pytest.approx(1.0)
    for i in range(5):
        calib.add_sample(make_sample(yaw=0.0, t_ms=i))
    assert calib.finished(1999) is False
    assert calib.finished(2000) is True


def test_low_fps_extends_collection_until_min_samples():
    calib = CenterCalibration(duration_s=3.0, min_samples=40, max_duration_s=10.0)
    calib.start(now_ms=0)
    for i in range(28):  # ~11 FPS for 3 s
        calib.add_sample(make_sample(yaw=0.05, t_ms=i))
    assert calib.finished(3000) is False  # keeps collecting past duration_s
    for i in range(12):
        calib.add_sample(make_sample(yaw=0.05, t_ms=3000 + i))
    assert calib.finished(4100) is True
    assert calib.finish().ok


def test_max_duration_gives_up_with_too_few_samples():
    calib = CenterCalibration(duration_s=3.0, min_samples=40, max_duration_s=10.0)
    calib.start(now_ms=0)
    calib.add_sample(make_sample(yaw=0.0, t_ms=0))
    assert calib.finished(9999) is False
    assert calib.finished(10000) is True
    result = calib.finish()
    assert not result.ok  # only 1 sample < min 40


def test_save_and_load_roundtrip(tmp_path):
    calib = CenterCalibration(duration_s=1.0, min_samples=5)
    calib.start(now_ms=0)
    for i in range(20):
        calib.add_sample(make_sample(yaw=0.07, pitch=0.01, iris_x=0.02, x=0.03, t_ms=i))
    result = calib.finish()
    assert result.ok and result.baseline is not None
    path = tmp_path / "baseline.json"
    save_baseline(path, result.baseline)

    loaded = load_baseline(path)
    assert math.isclose(loaded.yaw, result.baseline.yaw)
    assert math.isclose(loaded.std_x, result.baseline.std_x)
    assert loaded.n_samples == result.baseline.n_samples


def test_load_missing_and_corrupt_file(tmp_path):
    with pytest.raises(CalibrationError):
        load_baseline(tmp_path / "does_not_exist.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_baseline(bad)
    short = tmp_path / "short.json"
    short.write_text(
        '{"yaw":0,"pitch":0,"iris_x":0,"iris_y":0,"std_yaw":0.01,"std_iris_x":0.01,'
        '"std_x":0.01,"n_samples":3,"created_at":"x","version":1}',
        encoding="utf-8",
    )
    with pytest.raises(CalibrationError):  # too few samples
        load_baseline(short)
