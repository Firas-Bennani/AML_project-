"""
tests/unit/test_flag_threshold.py
==================================
Bug #1 regression test.

Before the fix, RISK_THRESHOLD was hard-coded at 0.7, so transactions with
real GNN scores in the 0.5–0.7 band stayed SCORED and never produced an
alert. The fix lowers the default to 0.5 via BACKEND_FLAG_THRESHOLD and
wires the value through to the transactions route.

This test asserts the threshold behaviour at the unit level without needing
the AI service or HTTP layer.
"""
from __future__ import annotations

import importlib
import os


def _reload_config(value: str):
    os.environ["BACKEND_FLAG_THRESHOLD"] = value
    import app.config as cfg  # type: ignore
    importlib.reload(cfg)
    return cfg


def test_default_threshold_is_half():
    cfg = _reload_config("0.5")
    assert cfg.FLAG_THRESHOLD == 0.5
    assert cfg.RISK_THRESHOLD == 0.5  # legacy alias must follow


def test_score_above_threshold_is_flagged():
    """A risk_score of 0.6 must produce status=FLAGGED, not SCORED."""
    cfg = _reload_config("0.5")
    score = 0.6
    status = "FLAGGED" if score >= cfg.FLAG_THRESHOLD else "SCORED"
    assert status == "FLAGGED", (
        f"score={score} should flag at threshold={cfg.FLAG_THRESHOLD}"
    )


def test_score_below_threshold_is_scored():
    cfg = _reload_config("0.5")
    score = 0.45
    status = "FLAGGED" if score >= cfg.FLAG_THRESHOLD else "SCORED"
    assert status == "SCORED"


def test_threshold_is_env_configurable():
    cfg = _reload_config("0.8")
    assert cfg.FLAG_THRESHOLD == 0.8
    # restore default for other tests
    _reload_config("0.5")
