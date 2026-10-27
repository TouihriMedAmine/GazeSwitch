"""Short audio cues for calibration steps.

Calibration asks the user to look at monitors that may not show the
preview, so progress has to be audible. Beeps play on a worker thread so the
tracking loop never blocks.
"""
from __future__ import annotations

import sys
import threading

_CUES = {
    "collect": [(880, 90)],  # start moving your head now
    "step": [(1320, 70)],  # this monitor is done, move on
    "done": [(660, 90), (990, 160)],  # rising: success
    "fail": [(520, 120), (390, 220)],  # falling: try again
}


def cue(kind: str) -> None:
    if sys.platform != "win32" or kind not in _CUES:
        return

    def play() -> None:
        import winsound

        for freq, ms in _CUES[kind]:
            try:
                winsound.Beep(freq, ms)
            except RuntimeError:
                return

    threading.Thread(target=play, name="sound-cue", daemon=True).start()
