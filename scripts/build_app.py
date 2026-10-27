"""Build the portable desktop app: dist/GazeSwitch/GazeSwitch.exe + zip.

Run from the project root with the project venv:
    python scripts/build_app.py

Requires: pip install pyinstaller pystray Pillow
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP = "GazeSwitch"
DIST = ROOT / "dist"
ICON = ROOT / "build" / "gazeswitch.ico"
ICON_PNG = ROOT / "assets" / "icon.png"
MODEL = ROOT / "models" / "face_landmarker.task"
USER_FILES = ("config.yaml", "baselines", "logs")


def build_icon() -> None:
    from PIL import Image

    ICON.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ICON_PNG).save(
        ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )


def main() -> int:
    if not MODEL.exists():
        print(f"Model missing: {MODEL}\nRun: python scripts/download_model.py")
        return 1
    build_icon()

    # PyInstaller wipes dist/GazeSwitch; keep the user's settings and data.
    app_folder = DIST / APP
    keep = ROOT / "build" / "user-data"
    shutil.rmtree(keep, ignore_errors=True)
    for name in USER_FILES:
        src = app_folder / name
        if src.exists():
            keep.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, keep / name)

    import PyInstaller.__main__

    sep = ";"  # Windows add-data separator
    PyInstaller.__main__.run([
        str(ROOT / "main.py"),
        "--name", APP,
        "--noconfirm",
        "--clean",
        "--windowed",
        "--icon", str(ICON),
        "--distpath", str(DIST),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        "--collect-all", "mediapipe",
        "--hidden-import", "pystray._win32",
        "--hidden-import", "scripts.list_cameras",
        # matplotlib must stay: mediapipe.tasks.python.vision imports it eagerly.
        "--exclude-module", "tkinter",
        "--exclude-module", "pytest",
        "--add-data", f"{MODEL}{sep}models",
        "--add-data", f"{ICON_PNG}{sep}assets",
        "--add-data", f"{ROOT / 'config.yaml'}{sep}.",
    ])

    archive = shutil.make_archive(str(DIST / f"{APP}-portable"), "zip", DIST, APP)

    if keep.exists():
        for item in keep.iterdir():
            dest = app_folder / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"Restored user data: {', '.join(p.name for p in keep.iterdir())}")
    print(f"\nApp folder: {app_folder}\nPortable zip: {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
