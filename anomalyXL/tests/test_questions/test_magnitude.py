from __future__ import annotations

import numpy as np

from anomalyxl.questions.magnitude import Magnitude


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


_BINS = {
    "Smallest": (1.0, 3.0),
    "Small": (3.0, 10.0),
    "Medium": (10.0, 30.0),
    "Large": (30.0, 100.0),
}


def test_basic_options_shape():
    tpl = Magnitude()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert len(row.options) == 5
        assert len(set(row.options)) == 5
        assert row.answer in row.options
        assert "No Anomaly" in row.options


def test_no_anomaly_has_no_injection():
    tpl = Magnitude()
    for seed in range(200):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        if row.answer == "No Anomaly":
            assert row.injections == []
            return
    raise AssertionError("No No-Anomaly gold seen")


def test_anomaly_gold_magnitude_in_bin():
    tpl = Magnitude()
    for seed in range(80):
        row = tpl.build(length=256, n_channels=1, rng=_rng(seed))
        if row.answer == "No Anomaly":
            continue
        lo, hi = _BINS[row.answer]
        assert len(row.injections) == 1
        m = row.injections[0].magnitude
        assert lo <= m < hi


def test_anomaly_kind_is_level_shift():
    tpl = Magnitude()
    for seed in range(40):
        row = tpl.build(length=256, n_channels=1, rng=_rng(seed))
        if row.injections:
            assert row.injections[0].kind == "level_shift"


def test_semantic_class_is_identity():
    tpl = Magnitude()
    assert tpl.semantic_class("Smallest") == "Smallest"
    assert tpl.semantic_class("No Anomaly") == "No Anomaly"
