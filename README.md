# GazeSwitch — Webcam-Based Multi-Monitor Targeting System

GazeSwitch watches you through a webcam (Camo virtual camera or any webcam),
estimates which of your three monitors you are looking at (`LEFT`, `CENTER`,
`RIGHT`, or `UNKNOWN`), and reports which application window on that monitor
would be the intended target for keyboard shortcuts.

**From source it runs in observe-only mode: it never changes window focus
until you explicitly arm targeting** (config gate + CLI flag + runtime
hotkey). The tray app (section 0) needs only the config gate.

```
camera -> face landmarks (MediaPipe) -> gaze features (head + iris)
       -> three-monitor calibration (learned head+eye model) -> LEFT/CENTER/RIGHT/UNKNOWN
       -> temporal smoothing + dwell + hysteresis -> stable selection
       -> Win32 window->monitor mapping -> candidate application
       -> observe-only reporting  OR  (armed) safe focus targeting
```

---

## 0. Desktop app (GazeSwitch.exe)

GazeSwitch also ships as a portable Windows tray app — no Python needed on
the target machine.

**Build** (from the project venv):

```powershell
python -m pip install pyinstaller pystray Pillow
python scripts\build_app.py
```

Output: `dist\GazeSwitch\` (run `GazeSwitch.exe`) and
`dist\GazeSwitch-portable.zip` (copy/unzip anywhere). From source, the same
app runs with `pythonw main.py --tray`.

**Using it:** an eye icon appears in the system tray (click the `^` arrow
near the clock if hidden; drag it onto the taskbar to keep it visible).

| Icon colour | Meaning |
|---|---|
| green | switching ON — looking at a monitor brings its window to the front |
| blue | watching only |
| orange | needs calibration |

Right-click menu: *Switch windows by gaze* (also double-click the icon or
Ctrl+Alt+G; Ctrl+Alt+K always stops it), *Calibrate*, *Show camera preview*,
*Silent mode*, *Camera* (Automatic or any detected camera by name; *Refresh
list* after plugging one in), *Start with Windows*, *Open settings file*,
*Open logs folder*, *Quit*. Menu items show their shortcut.

**Shortcuts** work in every app (change them under `hotkeys:` in
`config.yaml`; `""` turns one off):

| Shortcut | Action | In the preview |
|---|---|---|
| Ctrl+Alt+G | Switching on / off | T |
| Ctrl+Alt+K | Switching off, always | E |
| Ctrl+Alt+P | Show / hide the camera preview | Q or Esc hides |
| Ctrl+Alt+S | Silent mode: no notifications or beeps (saved) | S |
| Ctrl+Alt+C | Calibrate | C |
| Ctrl+Alt+N | Next camera (Automatic, then each camera) | N |
| Ctrl+Alt+Q | Quit GazeSwitch | |

In the preview, click the camera name at the top to pick a camera from a
list. Calibrating in silent mode opens the preview, since there are no beeps
to guide you.

- The camera choice is saved by device name (`camera.name` in
  `config.yaml`), so it survives index changes when devices come and go.
  Recalibrate if the new camera sits somewhere else.
- The preview stays above other windows (`overlay.always_on_top`), so
  switching focus on its monitor never buries it; it hides only with Q or
  its close button.

- On first run (no saved calibration) the preview opens: choose *Calibrate*
  (or press C) and follow the beeps (center, left, right; see §5). The
  preview closes itself on
  success; afterwards the app starts silently in the tray.
- `config.yaml`, `baselines\` and `logs\` live next to `GazeSwitch.exe`, so
  the folder is self-contained. Edit `config.yaml`, then quit and restart.
- In tray mode `targeting.enabled: true` is enough to allow switching (no
  `--enable-targeting` flag); `targeting.arm_on_start` turns it on at launch.
- Only one instance runs at a time; closing the preview window just hides it.

## 1. Installation (Windows PowerShell)

```powershell
cd "D:\Projects\AI Projects\Gaze_Tracker"
python -m pip install --disable-pip-version-check -r requirements.txt
python scripts\download_model.py      # fetches face_landmarker.task (~3.8 MB)
python -m pytest tests\ -q            # verify: 77 tests should pass
```

Verified working versions on this machine (Python 3.12.0):
`mediapipe==1.1.1`, `opencv-contrib-python==5.0.0.93`, `numpy 2.2.6`.

> **Note:** mediapipe declares `opencv-contrib-python` as its dependency. If
> you previously had plain `opencv-python` installed, remove it to avoid two
> conflicting cv2 copies: `pip uninstall opencv-python`.

## 2. Model asset setup

The MediaPipe **Tasks** Face Landmarker API requires a separate model file
that is NOT bundled with the pip package. `scripts\download_model.py` fetches
the official `face_landmarker.task` (float16, latest) from Google's MediaPipe
model storage into `models\face_landmarker.task`. If automatic download fails
(proxy/firewall), download it manually from the URL printed by the script and
place it at that path.

## 3. Identify the Camo camera index

```powershell
python main.py --list-cameras
```

On this machine (probed live): index `0` = 640x480 (built-in webcam),
**index `1` = 3840x2160 (Camo — the phone camera)**. Set it in `config.yaml`:

```yaml
camera:
  index: 1
```

or pass `--camera 1` on the command line. Camo must be running (Camo Studio
on the PC + Camo app on the phone) before starting GazeSwitch.

## 4. Running

```powershell
python main.py                        # observe-only (default, safe)
python main.py --camera 1             # explicit camera index
python main.py --no-preview           # console status only (no video window)
python main.py --probe                # print live x/yaw/iris values for tuning
python main.py --list-monitors        # show monitor->slot mapping + windows
python main.py --list-cameras         # probe camera indices
python main.py --config my.yaml       # alternative config file
```

On startup the app:
1. sets per-monitor-V2 DPI awareness (physical-pixel coordinates everywhere),
2. enumerates your monitors and maps them to slots by geometric x-order
   (left-most = LEFT, middle = CENTER, right-most = RIGHT),
3. optionally loads the saved center baseline from
   `baselines\center_baseline.json`,
4. opens the camera (Media Foundation first — with Camo at 4K it delivers
   ~29 fps vs ~9 fps for DirectShow; override with `camera.backend`) and
   reads it on a background thread that keeps only the newest frame,
5. opens the preview window (`GazeSwitch`).

> The preview window takes keyboard focus **once** when created at startup;
> click back into your work application afterwards. The app never re-steals
> focus after that.

## 5. Calibration

Press **C** in the preview (or tray menu → *Calibrate*). Calibration visits
your **center, left, then right** monitor (about 30 s in total). The
preview switches to a Face ID-style screen: your face in a circle with a
ring of ticks around it.

1. Move your eyes to the monitor it names. A countdown gives you 3 s
   (`settle_s`).
2. At the **beep**, keep your eyes on that monitor and **slowly move your
   head in a circle**. Ticks turn green for each direction you've reached.
   A full ring ends the step after at least ~5 s; otherwise it ends after
   8 s (`duration_s`). A short high beep means move on to the next monitor.
3. After the right monitor a rising chime means success (falling tone =
   try again; the message says which monitor it couldn't see you at).

Turning your head while your eyes stay put is what teaches GazeSwitch how
much your eyes make up for your head rotation. Afterwards, *head turned
left but eyes on the center monitor* counts as CENTER instead of LEFT.
Window switching pauses while calibration runs.

The result (`baselines\center_baseline.json`) auto-loads on future runs.
Baselines from older versions (center only) still work with the fixed
weights below; the preview asks you to recalibrate.

Recalibrate any time with **C** (also after moving the camera, changing
posture/seat position, or if side detection drifts).

**Learned model:** calibration fits `raw = w_yaw*yaw + w_iris*iris + bias`
(regularised least squares on all three monitors' samples) and stores each
monitor's median, so x is exactly -1 / 0 / +1 at your left / center / right
monitor. Boundaries sit at ±0.5 with an undecided band of
`classifier.monitor_band` (default 0.1) either side; the Details panel (D)
shows `center ≤ 0.40 side ≥ 0.60`. The log line `Calibration OK` records
the weights, anchors and how many calibration samples were classified
correctly.

**Fixed weights (no learned model):** the classifier compares the *current*
combined signal (weighted head-yaw + relative-iris-x, both relative to the
baseline) against two thresholds:

- `|x| <= center_threshold` (default 0.12) → `CENTER`
- `|x| >= side_threshold` (default 0.32) → `LEFT` / `RIGHT`
- in between → `UNKNOWN` (deliberate hysteresis band)

**Tuning:** run `python main.py --probe`, look toward each monitor, note the
printed `x` values, then set `classifier.center_threshold` /
`classifier.side_threshold` in `config.yaml` (or enable
`auto_thresholds_from_noise: true` to derive them from calibration noise).
If LEFT/RIGHT come out swapped on your setup, set `classifier.invert_x: true`.

## 6. Observe-only mode (default)

The preview shows the camera full-bleed with floating glass controls:

- **top left** — mode: *Switching on* (green), *Watching only* (grey) or
  *Not calibrated* (orange);
- **top right** — key hints: `C` Calibrate, `D` Details, `Q` Quit (Hide in
  the tray app); `T` toggles switching and `E` stops it;
- **bottom card** — the monitor you're looking at and the window that would
  be focused, three monitor glyphs (solid blue = selected, blue outline =
  settling) and a dot showing the live gaze signal;
- **Details** (`D`) — raw `x`/`yaw`/`iris`, thresholds, votes, FPS and
  camera status, plus extra face landmarks.

During calibration the video dims behind a progress card. Messages (e.g.
"Switching off", "Calibrated") appear briefly as a capsule under the top
bar.

Monitor transitions and target-window changes are logged to the console and
`logs\gazeswitch.log`. No focus is changed; no input is injected — ever — in
this mode.

## 7. Targeting mode (optional, Phase 2)

Three switches must all be on:

1. `config.yaml`: `targeting.enabled: true`
2. CLI: `python main.py --enable-targeting`
3. Runtime: global hotkey **Ctrl+Alt+G** (or `T` in the preview) — the
   preview's mode capsule turns green: *Switching on*.

When armed, after a *stable* monitor switch (vote window + dwell already
satisfied) **and** the candidate window being unchanged for
`target_dwell_ms` (150 ms), the app brings the candidate window to the
foreground, with guards:

- **No focus steal while typing**: if any keyboard/mouse input happened in the
  last `typing_quiet_ms` (500 ms, `GetLastInputInfo`), the switch waits.
- **Foreground lock**: Windows only lets the process that received the last
  input take the foreground, so a plain `SetForegroundWindow` from a
  background app is refused (the taskbar button just flashes). GazeSwitch
  then injects one *unassigned* virtual key (0xE8, ignored by applications),
  attaches to the foreground thread's input queue and retries. The result is
  verified; failures are logged with a 2 s cooldown.
- **Emergency stop**: **Ctrl+Alt+K** (global) or `E` in the preview disarms
  instantly at any time.

Once focused, each app restores its own last keyboard focus (e.g. the
YouTube player, Explorer's file list, Cursor's chat box) — GazeSwitch never
clicks inside windows.

**Latency** with the shipped settings at ~30 fps: about 0.5–0.6 s from your
eyes settling on a monitor to its window being focused (~150 ms vote
majority in `vote_window_ms: 250`, `dwell_ms: 200`, `target_dwell_ms: 150`).
Lower these for faster switching at the cost of more accidental switches on
quick glances; `python scripts\benchmark.py` shows camera vs detection speed.

## 8. Troubleshooting incorrect monitor detection

| Symptom | Fix |
|---|---|
| LEFT/RIGHT swapped | Set `classifier.invert_x: true`, recalibrate |
| Side monitors unreachable (stays CENTER) | Lower `side_threshold` (probe first: `--probe`) |
| Random flips between CENTER and a side | Raise `side_threshold`, raise `dwell_ms` / `window_frames` |
| UNKNOWN most of the time | Not calibrated → press C; or thresholds too far apart; check the preview doesn't say *Can't see your face* |
| Wrong monitor arrangement | Check `python main.py --list-monitors`; slot order = x-order of monitor rects |
| Head-only movement triggers | Lower `weight_head` / raise `weight_iris` in `classifier` |
| Eyes-only noise triggers | Opposite of the above |
| Camera disconnects | App auto-reopens (status in panel); check Camo Studio is running |
| Wrong candidate app on a monitor | Set `windows.preferences` per slot in config.yaml (explicit mapping) |
| Camo window itself listed | Add its process name to `windows.excluded_processes` |

## 9. Configuration reference (config.yaml)

All keys optional; defaults shown in the file. Sections: `camera` (index,
size, mirror, downscale), `tracking` (model path, confidences, head-pose
method, sign inversions), `calibration` (duration, min samples, save path,
auto-load), `classifier` (weights, scales, thresholds, invert_x),
`selection` (window_frames, min_confidence, dwell_ms, lost_ms), `windows`
(refresh, exclusions, per-slot preferences), `targeting` (master gate, typing
guard, dwell, backoff), `hotkeys`, `overlay`, `logging`.

## 10. Architecture

```
main.py                 orchestration: camera loop -> pipeline -> actions
src/
  camera.py             resilient capture (DirectShow/MSMF), reconnect,
                        early downscale (Camo 4K -> 640), centralised mirroring
  face_tracker.py       MediaPipe Tasks FaceLandmarker (VIDEO mode) ->
                        FaceObservation (yaw/pitch/roll, iris offsets, blink)
  head_pose (in face_tracker)   geometric ratios (default) or matrix Euler;
                        both approximate — used only *relative to baseline*
  gaze_features.py      baseline-relative combined x/y signal (weights)
  calibration.py        three-monitor calibration + head/eye model fit, save/load
  monitor_classifier.py LEFT/CENTER/RIGHT/UNKNOWN thresholds + hysteresis band
  selection_state.py    vote window + confidence + dwell + lost-tracking
  window_manager.py     Win32 ctypes: DPI awareness, monitors, windows,
                        largest-intersection monitor assignment
  target_manager.py     candidate selection per slot (preferences > area > z)
  action_router.py      transition callbacks, single-fire dedupe
  targeting.py          armed focus controller (guards, verify, backoff)
  input_hotkeys.py      RegisterHotKey thread for the global shortcuts
  overlay.py            preview diagnostics rendering
  tray.py               system-tray icon + menu (pystray), request queue
  app_env.py            portable paths, single instance, start with Windows
  config.py / logger.py typed config loading, logging setup
scripts/build_app.py    PyInstaller build -> dist\GazeSwitch + portable zip
scripts/benchmark.py    camera FPS vs face-detection time
tests/                  77 unit tests (no camera or GUI required)
```

Sign conventions (documented, consistent): frames are processed mirrored
(selfie view); positive x = user looking toward **their right**. Head
`yaw`/`pitch` are *pseudo-radian proxies* — the app deliberately does not
claim calibrated physical angles from an ordinary webcam.

## 11. Known limitations & safety

- **Accuracy is heuristic.** Monitor-level selection from a webcam is
  realistic; pixel-precise gaze is not. Measure it on your own setup with
  `scripts\evaluate_gaze.py` (§13).
- **The learned model is linear.** It handles monitors at uneven angles
  (each side has its own anchor) and head/eye trade-off, but not large
  posture changes since calibration — recalibrate after moving your seat.
- **Posture sensitivity.** Baselines drift when you lean/move the camera;
  recalibrate (C) when detection degrades.
- **Blinks** are filtered via blendshapes; extremely fast glances are
  intentionally suppressed by the vote window + dwell.
- **Windows foreground lock** can refuse focus changes (see §7).
- **Privacy:** all processing is local; no frames or biometric data leave
  the machine. The only network access is the one-time model download.

## 12. Future extension: within-monitor gaze position (design, not implemented)

The architecture leaves a clean path for approximate `(x, y)` gaze on the
selected monitor:

1. **Multi-point calibration** — extend `CenterCalibration` to per-monitor
   anchor points (5–9 points each): show a marker, press a key, store the
   feature vector. `GazeBaseline` already stores per-feature mean/std; store
   one per anchor instead.
2. **Feature vector** — `GazeSample` already carries baseline-relative
   `x, y, yaw_rel, pitch_rel, iris_dx, iris_dy`. A per-monitor affine
   (or thin-plate) map from these features to that monitor's pixel
   coordinates is the next layer, kept *per monitor* because each screen
   has a different angle to the camera.
3. **Per-monitor transforms** — store one transform + its residual error per
   monitor; `MonitorInfo` already carries physical rect + DPI, so mapping to
   physical pixels is already DPI-correct.
4. **Quality reporting** — report calibration residual (mean |error| in px)
   and refuse to act when residual is above a threshold.
5. **Honest expectations** — webcam point-of-gaze is typically accurate to
   tens of pixels at best, and only for the calibrated user/posture; use it
   for "which window/card on this monitor", not for pixel-level pointing.

The classifier/selection/action interfaces (states, events, callbacks) do
not need to change to add this.

## 13. Measuring accuracy and latency

```powershell
python scripts\evaluate_gaze.py                 # 10 looks per monitor, ~1.5 min
python scripts\evaluate_gaze.py --per-monitor 15
python scripts\evaluate_gaze.py --log-only      # focus latency from the app log only
```

Quit GazeSwitch first (it holds the camera) and calibrate beforehand. A
"Look here" window jumps between your monitors in random order (never the
same monitor twice in a row). The script runs the app's own pipeline and
records, per cue, the monitor selected after 2.5 s and the time until the
right monitor was selected. That time includes your own eye and head
reaction (typically 0.2–0.3 s). Focus latency, from a selection to the
window being focused, is read from `logs\gazeswitch.log` of normal use; its
tail includes deliberate delays such as the typing guard
(`targeting.typing_quiet_ms`).

It prints a confusion table and a metrics table in Markdown and saves
`logs\eval-*.json`. Results hold for one user, one desk and one camera.

## License

MIT, see [LICENSE](LICENSE). The MediaPipe face model is not included; it
is downloaded from Google under its own license (Apache 2.0).
