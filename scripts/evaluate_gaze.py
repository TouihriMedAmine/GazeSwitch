"""Measure GazeSwitch's monitor accuracy and latency on your own setup.

A "Look here" window jumps between your monitors in random order. For each
cue the script runs the same pipeline as the app (camera, face tracker,
learned gaze model, vote + dwell selection) and records:

* the monitor selected 2.5 s after the cue (accuracy, confusion table),
* the time from the cue until the right monitor is selected (includes your
  own eye/head reaction time, typically 0.2-0.3 s),
* from the app's log: the time from a selection to the window being focused.

Close GazeSwitch first (only one program can use the camera), and calibrate
beforehand. Run from the project root:

    python scripts\\evaluate_gaze.py              # 10 looks per monitor
    python scripts\\evaluate_gaze.py --per-monitor 15
    python scripts\\evaluate_gaze.py --log-only   # focus latency from the log only

Results print as Markdown and are saved to <app dir>/logs/eval-*.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLOTS = ("LEFT", "CENTER", "RIGHT")
DETECTED = SLOTS + ("UNKNOWN",)


# ------------------------------------------------------------ pure helpers
def make_trial_order(per_monitor: int, rng: random.Random, start: str = "CENTER") -> List[str]:
    """Each monitor ``per_monitor`` times, shuffled, never the same twice in a
    row and never ``start`` first (where the user looks before the first cue):
    a repeat needs no eye movement and would inflate the results."""
    remaining = {s: per_monitor for s in SLOTS}
    order: List[str] = []
    while any(remaining.values()):
        previous = order[-1] if order else start
        options = [s for s, n in remaining.items() if n and s != previous]
        if not options:  # only the previous monitor is left: restart the draw
            return make_trial_order(per_monitor, rng, start)
        weights = [remaining[s] for s in options]
        pick = rng.choices(options, weights=weights)[0]
        order.append(pick)
        remaining[pick] -= 1
    return order


_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*?(Monitor entered:|"
                       r"Targeting: focused window)")


def focus_latencies_from_log(lines: Sequence[str]) -> List[float]:
    """Milliseconds from 'Monitor entered' to the next 'focused window' line
    (only when the focus happens before the selection changes again)."""
    latencies: List[float] = []
    entered: Optional[datetime] = None
    for line in lines:
        m = _LOG_LINE.match(line)
        if not m:
            continue
        stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        if m.group(2) == "Monitor entered:":
            entered = stamp
        elif entered is not None:
            latencies.append((stamp - entered).total_seconds() * 1000.0)
            entered = None
    return latencies


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(trials: List[dict], focus_ms: List[float]) -> dict:
    confusion = {t: {d: 0 for d in DETECTED} for t in SLOTS}
    for trial in trials:
        confusion[trial["target"]][trial["detected"]] += 1
    correct = sum(confusion[s][s] for s in SLOTS)
    latencies = [t["latency_ms"] for t in trials if t["latency_ms"] is not None]
    return {
        "trials": len(trials),
        "correct": correct,
        "accuracy": correct / len(trials) if trials else float("nan"),
        "confusion": confusion,
        "selection_latency_ms": {
            "n": len(latencies),
            "median": statistics.median(latencies) if latencies else float("nan"),
            "p90": percentile(latencies, 0.9),
        },
        "focus_latency_ms": {
            "n": len(focus_ms),
            "median": statistics.median(focus_ms) if focus_ms else float("nan"),
            "p90": percentile(focus_ms, 0.9),
        },
    }


def to_markdown(summary: dict, setup: str) -> str:
    lines = [f"Setup: {setup}", ""]
    if summary["trials"]:
        lines += ["| Looked at \\ Detected | " + " | ".join(d.title() for d in DETECTED) + " |",
                  "|---|" + "---:|" * len(DETECTED)]
        for t in SLOTS:
            row = summary["confusion"][t]
            lines.append(f"| {t.title()} | " + " | ".join(str(row[d]) for d in DETECTED) + " |")
        sel = summary["selection_latency_ms"]
        lines += ["", "| Metric | Result |", "|---|---:|",
                  f"| Monitor accuracy | {summary['correct']} / {summary['trials']} "
                  f"({summary['accuracy']:.0%}) |",
                  f"| Cue → monitor selected (median, p90; n={sel['n']}) | "
                  f"{sel['median'] / 1000:.2f} s, {sel['p90'] / 1000:.2f} s |"]
    else:
        lines += ["| Metric | Result |", "|---|---:|"]
    foc = summary["focus_latency_ms"]
    if foc["n"]:
        lines.append(f"| Selection → window focused (median, p90; n={foc['n']}) | "
                     f"{foc['median']:.0f} ms, {foc['p90']:.0f} ms |")
    return "\n".join(lines)


# --------------------------------------------------------------- live run
def _running_gazeswitch() -> bool:
    import subprocess

    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq GazeSwitch.exe"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GazeSwitch.exe" in out


def _cue_image(text: str, sub: str):
    import cv2
    import numpy as np

    img = np.zeros((320, 520, 3), np.uint8)
    cv2.circle(img, (260, 130), 46, (255, 132, 10), -1, cv2.LINE_AA)
    cv2.circle(img, (260, 130), 10, (255, 255, 255), -1, cv2.LINE_AA)
    for line, y, scale in ((text, 230, 0.9), (sub, 270, 0.55)):
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, line, ((520 - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (235, 235, 235), 2, cv2.LINE_AA)
    return img


def run_session(app_dir: Path, per_monitor: int, hold_s: float, seed: Optional[int]) -> List[dict]:
    import cv2

    from main import GazeSwitchApp, parse_args
    from src.calibration import CalibrationError, load_baseline
    from src.camera_devices import list_cameras
    from src.config import load_config
    from src.face_tracker import FaceTracker
    from src.overlay import style_window

    cfg = load_config(str(app_dir / "config.yaml"))
    app = GazeSwitchApp(cfg, parse_args(["--config", str(app_dir / "config.yaml")]))
    try:
        baseline = load_baseline(app_dir / cfg.calibration.save_path)
    except CalibrationError as exc:
        sys.exit(f"No calibration found ({exc}). Calibrate in GazeSwitch first.")
    if baseline.model is None:
        print("Note: this calibration has no learned head/eye model; recalibrate for "
              "results that match the current app.")
    app.baseline = baseline
    app.classifier.set_baseline(baseline)
    tracker = FaceTracker(
        model_path=str(ROOT / cfg.tracking.model_path),
        min_detection_confidence=cfg.tracking.min_detection_confidence,
        min_presence_confidence=cfg.tracking.min_presence_confidence,
        min_tracking_confidence=cfg.tracking.min_tracking_confidence,
        head_pose_method=cfg.tracking.head_pose_method,
        blink_threshold=cfg.tracking.blink_threshold,
        invert_yaw=cfg.tracking.invert_yaw,
        invert_pitch=cfg.tracking.invert_pitch,
        blendshapes=cfg.tracking.blendshapes,
    )
    app.camera_devices = list_cameras()
    if not app._open_camera(cfg.camera.name, cfg.camera.index):
        sys.exit("No working camera found.")
    print(f"Camera: {app.camera_label}")

    order = make_trial_order(per_monitor, random.Random(seed))
    window = "GazeSwitch evaluation"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def show(slot: str, text: str, sub: str) -> None:
        x0, y0, x1, y1 = app.slots[slot].rect
        cv2.imshow(window, _cue_image(text, sub))
        cv2.moveWindow(window, (x0 + x1) // 2 - 260, (y0 + y1) // 2 - 180)
        style_window(window, topmost=True)
        cv2.waitKey(1)

    def step(now_ms: int):
        ok, frame = app.camera.read()
        if not ok or frame is None:
            return None
        scale = min(1.0, cfg.camera.process_width / max(1, frame.shape[1]))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        obs = tracker.detect(frame, now_ms)
        sample = app.extractor.extract(obs, app.baseline, now_ms)
        cls = app.classifier.classify(sample)
        return app.state_machine.update(cls, obs is not None, now_ms)

    show("CENTER", "Get ready", "Follow the dot with your eyes, as you normally would")
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        step(int(time.monotonic() * 1000))
        cv2.waitKey(1)

    trials: List[dict] = []
    try:
        for i, target in enumerate(order, start=1):
            show(target, "Look here", f"{i} / {len(order)}")
            t0 = int(time.monotonic() * 1000)
            latency: Optional[float] = None
            event = None
            while (now := int(time.monotonic() * 1000)) - t0 < hold_s * 1000:
                result = step(now)
                if result is not None:
                    event = result
                    if latency is None and event.stable and event.state.value == target:
                        latency = float(now - t0)
                cv2.waitKey(1)
            detected = event.state.value if event is not None and event.stable else "UNKNOWN"
            trials.append({"target": target, "detected": detected, "latency_ms": latency})
            mark = "ok " if detected == target else "MISS"
            print(f"  {i:2d}/{len(order)} {mark} looked {target:6} detected {detected:7} "
                  f"latency {'-' if latency is None else f'{latency / 1000:.2f}s'}")
    finally:
        cv2.destroyAllWindows()
        app.camera.release()
        tracker.close() if hasattr(tracker, "close") else None
    return trials


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_dir = ROOT / "dist" / "GazeSwitch"
    parser.add_argument("--app-dir", type=Path,
                        default=default_dir if (default_dir / "config.yaml").exists() else ROOT,
                        help="Folder with the config.yaml / baselines / logs to use "
                             "(default: the built app, else the project)")
    parser.add_argument("--per-monitor", type=int, default=10)
    parser.add_argument("--hold", type=float, default=2.5,
                        help="Seconds per cue; the monitor selected at the end counts")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-only", action="store_true",
                        help="Only compute focus latency from the app log")
    args = parser.parse_args(argv)

    trials: List[dict] = []
    if not args.log_only:
        if _running_gazeswitch():
            sys.exit("GazeSwitch is running and holds the camera. Quit it (Ctrl+Alt+Q) "
                     "and run this again.")
        print(f"Using {args.app_dir}. {args.per_monitor * 3} cues, {args.hold:.1f} s each "
              f"(~{args.per_monitor * 3 * args.hold / 60:.1f} min).")
        input("Sit as you normally do, then press Enter to start...")
        trials = run_session(args.app_dir, args.per_monitor, args.hold, args.seed)

    log_path = args.app_dir / "logs" / "gazeswitch.log"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() \
        if log_path.exists() else []
    summary = summarize(trials, focus_latencies_from_log(lines))
    setup = f"{datetime.now():%Y-%m-%d}, one user, three monitors"
    print("\n" + to_markdown(summary, setup))

    out_dir = args.app_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "trials": trials}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
