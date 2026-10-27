"""Face tracking via MediaPipe Tasks Face Landmarker.

Produces a :class:`FaceObservation` per frame: approximate head pose
(yaw/pitch/roll), relative iris offsets, blink score and the landmark array
(for the preview overlay).

MediaPipe is imported lazily inside the constructor so this module (and every
module importing it) stays importable in headless test environments.

Head pose methods
-----------------
* ``geometric`` (default): transparent landmark ratios (nose offset relative
  to face centre/width). Approximate by definition, but stable and
  convention-safe.
* ``matrix``: Euler angles from the model's facial transformation matrix.
  Also approximate (no calibrated camera intrinsics) and its sign conventions
  can vary; ``invert_yaw`` / ``invert_pitch`` allow correction after checking
  the on-screen bars.

Neither method yields physically accurate angles from an ordinary webcam;
both are used only as *relative* features around the calibration baseline.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Canonical FaceMesh topology indices (478 landmarks incl. refined iris).
NOSE_TIP = 1
FACE_CONTOUR_A, FACE_CONTOUR_B = 234, 454  # outermost left/right contour points
EYE_A_OUTER, EYE_A_INNER, EYE_A_TOP, EYE_A_BOTTOM = 33, 133, 159, 145
EYE_B_OUTER, EYE_B_INNER, EYE_B_TOP, EYE_B_BOTTOM = 263, 362, 386, 374
IRIS_A_CENTER = 468
IRIS_B_CENTER = 473


class FaceTrackerError(RuntimeError):
    """Raised for setup problems the user can act on (missing model, etc.)."""


@dataclass
class FaceObservation:
    t_ms: int
    yaw: float = 0.0  # pseudo-radians; + = user looking toward their right
    pitch: float = 0.0  # pseudo-radians; + = looking down
    roll: float = 0.0
    iris_dx: float = 0.0  # relative iris x in [-0.5, 0.5]; + = toward image right
    iris_dy: float = 0.0  # relative iris y; + = iris lower in the eye
    blink: float = 0.0  # 0..1 blendshape-ish blink score
    face_width: float = 0.0
    face_height: float = 0.0
    landmarks: Optional[np.ndarray] = None  # (N, 3) normalised, for overlay
    method: str = "geometric"


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-6:
        return default
    return numerator / denominator


class FaceTracker:
    """Detect one face per frame and derive gaze features."""

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        head_pose_method: str = "geometric",
        blink_threshold: float = 0.5,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
        yaw_ratio_gain: float = 2.4,
        pitch_ratio_gain: float = 2.0,
        blendshapes: bool = False,
    ) -> None:
        if not Path(model_path).exists():
            raise FaceTrackerError(
                f"Face landmarker model not found at '{model_path}'.\n"
                "Run:  python scripts/download_model.py\n"
                "See README section 'Model asset setup'."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - environment issue
            raise FaceTrackerError(
                f"mediapipe could not be imported ({exc}). "
                "Run: pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            # Blendshapes run a second model; blink falls back to eye aspect ratio.
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=(head_pose_method == "matrix"),
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._last_ts = -1
        self.method = head_pose_method
        self.blink_threshold = blink_threshold
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self.yaw_ratio_gain = yaw_ratio_gain
        self.pitch_ratio_gain = pitch_ratio_gain
        self._consecutive_errors = 0
        log.info("FaceTracker initialised (method=%s, model=%s)", head_pose_method, model_path)

    # ------------------------------------------------------------------ API
    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> Optional[FaceObservation]:
        """Run detection. Returns None when no face is found; never raises on
        per-frame inference problems (errors are logged and counted)."""
        try:
            with self._lock:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb
                )
                ts = int(t_ms)
                if ts <= self._last_ts:  # MediaPipe requires strictly increasing ts
                    ts = self._last_ts + 1
                self._last_ts = ts
                result = self._landmarker.detect_for_video(mp_image, ts)
            self._consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001 - inference failures must not kill the app
            self._consecutive_errors += 1
            if self._consecutive_errors <= 3 or self._consecutive_errors % 100 == 0:
                log.error("FaceTracker inference failure (%d): %s", self._consecutive_errors, exc)
            return None

        if not result.face_landmarks:
            return None
        landmarks = np.array(
            [[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32
        )
        return self._build_observation(landmarks, result, int(t_ms))

    def close(self) -> None:
        with self._lock:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
        log.info("FaceTracker closed")

    # ------------------------------------------------------------- internals
    def _build_observation(
        self, lm: np.ndarray, result: Any, t_ms: int
    ) -> FaceObservation:
        yaw = pitch = roll = 0.0
        if self.method == "matrix" and result.facial_transformation_matrixes:
            rot = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)[
                :3, :3
            ]
            angles_deg = cv2.RQDecomp3x3(rot)[1]
            yaw, pitch, roll = (
                np.deg2rad(angles_deg[1]),
                np.deg2rad(angles_deg[0]),
                np.deg2rad(angles_deg[2]),
            )
        else:
            yaw = self._geometric_yaw(lm)
            pitch = self._geometric_pitch(lm)

        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch

        iris_dx = self._iris_dx(lm)
        iris_dy = self._iris_dy(lm)
        blink = self._blink_score(result, lm)

        x_a, x_b = lm[FACE_CONTOUR_A, 0], lm[FACE_CONTOUR_B, 0]
        y_top, y_bottom = lm[10, 1], lm[152, 1]
        return FaceObservation(
            t_ms=t_ms,
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            iris_dx=float(iris_dx),
            iris_dy=float(iris_dy),
            blink=float(blink),
            face_width=float(abs(x_b - x_a)),
            face_height=float(abs(y_bottom - y_top)),
            landmarks=lm,
            method=self.method,
        )

    def _geometric_yaw(self, lm: np.ndarray) -> float:
        """Nose-tip offset from face centre, normalised by face width.

        In the mirrored frame a head turn toward the user's right moves the
        nose toward the image right, so positive ratio == user's right.
        """
        center_x = (lm[FACE_CONTOUR_A, 0] + lm[FACE_CONTOUR_B, 0]) / 2.0
        width = abs(lm[FACE_CONTOUR_B, 0] - lm[FACE_CONTOUR_A, 0])
        ratio = _safe_ratio(lm[NOSE_TIP, 0] - center_x, width)
        return ratio * self.yaw_ratio_gain

    def _geometric_pitch(self, lm: np.ndarray) -> float:
        center_y = (
            lm[EYE_A_TOP, 1] + lm[EYE_A_BOTTOM, 1] + lm[EYE_B_TOP, 1] + lm[EYE_B_BOTTOM, 1]
        ) / 4.0
        height = abs(lm[152, 1] - lm[10, 1])
        ratio = _safe_ratio(lm[NOSE_TIP, 1] - center_y, height)
        return ratio * self.pitch_ratio_gain

    @staticmethod
    def _iris_dx(lm: np.ndarray) -> float:
        """Average relative iris x offset across both eyes.

        Denominators are ordered so that 'positive' means the iris moved
        toward the image right in *both* eyes regardless of eye naming.
        """
        iris_a_x = lm[IRIS_A_CENTER, 0]
        mid_a = (lm[EYE_A_OUTER, 0] + lm[EYE_A_INNER, 0]) / 2.0
        width_a = lm[EYE_A_INNER, 0] - lm[EYE_A_OUTER, 0]  # inner is right of outer
        dx_a = _safe_ratio(iris_a_x - mid_a, width_a)

        iris_b_x = lm[IRIS_B_CENTER, 0]
        mid_b = (lm[EYE_B_OUTER, 0] + lm[EYE_B_INNER, 0]) / 2.0
        width_b = lm[EYE_B_OUTER, 0] - lm[EYE_B_INNER, 0]  # outer is right of inner
        dx_b = _safe_ratio(iris_b_x - mid_b, width_b)
        return (dx_a + dx_b) / 2.0

    @staticmethod
    def _iris_dy(lm: np.ndarray) -> float:
        iris_a_y = lm[IRIS_A_CENTER, 1]
        mid_a = (lm[EYE_A_TOP, 1] + lm[EYE_A_BOTTOM, 1]) / 2.0
        height_a = lm[EYE_A_BOTTOM, 1] - lm[EYE_A_TOP, 1]
        dy_a = _safe_ratio(iris_a_y - mid_a, height_a)

        iris_b_y = lm[IRIS_B_CENTER, 1]
        mid_b = (lm[EYE_B_TOP, 1] + lm[EYE_B_BOTTOM, 1]) / 2.0
        height_b = lm[EYE_B_BOTTOM, 1] - lm[EYE_B_TOP, 1]
        dy_b = _safe_ratio(iris_b_y - mid_b, height_b)
        return (dy_a + dy_b) / 2.0

    def _blink_score(self, result: Any, lm: np.ndarray) -> float:
        try:
            if result.face_blendshapes:
                scores = {
                    c.category_name: c.score for c in result.face_blendshapes[0]
                }
                left = float(scores.get("eyeBlinkLeft", 0.0))
                right = float(scores.get("eyeBlinkRight", 0.0))
                return max(left, right)
        except Exception:  # noqa: BLE001 - fall through to EAR estimate
            pass
        # Fallback: eye aspect ratio (open ~0.30, closed ~0.08).
        ear_a = _safe_ratio(
            abs(lm[EYE_A_BOTTOM, 1] - lm[EYE_A_TOP, 1]),
            abs(lm[EYE_A_INNER, 0] - lm[EYE_A_OUTER, 0]),
            default=0.3,
        )
        ear_b = _safe_ratio(
            abs(lm[EYE_B_BOTTOM, 1] - lm[EYE_B_TOP, 1]),
            abs(lm[EYE_B_INNER, 0] - lm[EYE_B_OUTER, 0]),
            default=0.3,
        )
        ear = (ear_a + ear_b) / 2.0
        return float(np.clip((0.22 - ear) / 0.12, 0.0, 1.0))
