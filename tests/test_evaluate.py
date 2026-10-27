"""Scoring helpers of scripts/evaluate_gaze.py (no camera)."""
from __future__ import annotations

import random

from scripts.evaluate_gaze import (focus_latencies_from_log, make_trial_order, percentile,
                                   summarize, to_markdown)


def test_trial_order_is_balanced_without_repeats():
    for seed in range(20):
        order = make_trial_order(10, random.Random(seed))
        assert sorted(order) == sorted(["LEFT", "CENTER", "RIGHT"] * 10)
        assert all(a != b for a, b in zip(order, order[1:]))
        assert order[0] != "CENTER"  # the "get ready" screen is on the center monitor


def test_focus_latency_pairs_selection_with_next_focus():
    log = [
        "2026-10-27 00:22:34,192 INFO    src.action_router: Monitor entered: LEFT (x)",
        "2026-10-27 00:22:34,347 INFO    src.targeting: Targeting: focused window hwnd=1",
        "2026-10-27 00:22:35,694 INFO    src.action_router: Monitor entered: CENTER (y)",
        "2026-10-27 00:22:36,000 INFO    gazeswitch.main: status: ...",
        "2026-10-27 00:22:36,370 INFO    src.targeting: Targeting: focused window hwnd=2",
        "2026-10-27 00:22:37,000 INFO    src.targeting: Targeting: focused window hwnd=3",
        "2026-10-27 00:22:38,000 INFO    src.action_router: Monitor entered: RIGHT (z)",
    ]
    assert focus_latencies_from_log(log) == [155.0, 676.0]


def test_summary_and_markdown():
    trials = [
        {"target": "LEFT", "detected": "LEFT", "latency_ms": 600.0},
        {"target": "CENTER", "detected": "CENTER", "latency_ms": 500.0},
        {"target": "RIGHT", "detected": "UNKNOWN", "latency_ms": None},
    ]
    s = summarize(trials, [150.0, 160.0])
    assert s["correct"] == 2 and s["confusion"]["RIGHT"]["UNKNOWN"] == 1
    assert s["selection_latency_ms"]["median"] == 550.0
    md = to_markdown(s, "test")
    assert "2 / 3 (67%)" in md and "155 ms" in md
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
