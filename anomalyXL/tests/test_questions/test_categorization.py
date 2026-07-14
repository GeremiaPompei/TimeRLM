from __future__ import annotations

import numpy as np

from anomalyxl.anomalies import ANOMALY_KINDS
from anomalyxl.questions.categorization import Categorization


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_basic_row_shape():
    tpl = Categorization()
    row = tpl.build(length=256, n_channels=2, rng=_rng(0))
    assert isinstance(row.options, list) and len(row.options) == 5
    assert row.answer in row.options
    assert len(row.series) == 2
    for arr in row.series.values():
        assert len(arr) == 256


def test_no_anomaly_carries_empty_injections():
    tpl = Categorization()
    # Force the gold by iterating until we hit a no-anomaly case.
    for seed in range(200):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        if row.answer == "No Anomaly":
            assert row.injections == []
            return
    raise AssertionError("Never sampled a No Anomaly gold across 200 seeds")


def test_anomaly_gold_carries_one_injection():
    tpl = Categorization()
    label_to_kind = {
        "Level Shift": "level_shift",
        "Transient Spike": "transient_spike",
        "Change in Seasonality": "change_in_seasonality",
        "Change in Variance": "change_in_variance",
        "Change in Trend": "change_in_trend",
    }
    for seed in range(50):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed))
        if row.answer in label_to_kind:
            assert len(row.injections) == 1
            assert row.injections[0].kind == label_to_kind[row.answer]
            assert row.injections[0].channel in row.series
            return
    raise AssertionError("Never sampled an anomaly gold across 50 seeds")


def test_options_always_include_no_anomaly():
    tpl = Categorization()
    for seed in range(50):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        assert "No Anomaly" in row.options


def test_options_have_no_duplicates():
    tpl = Categorization()
    for seed in range(50):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        assert len(set(row.options)) == 5


def test_semantic_class_bin_mapping():
    tpl = Categorization()
    assert tpl.semantic_class("Change in Seasonality") == "Change in Seasonality or Variance"
    assert tpl.semantic_class("Change in Variance") == "Change in Seasonality or Variance"
    assert tpl.semantic_class("Level Shift") == "Level Shift"
    assert tpl.semantic_class("No Anomaly") == "No Anomaly"


def test_seed_reproducibility():
    tpl = Categorization()
    a = tpl.build(length=128, n_channels=2, rng=_rng(7))
    b = tpl.build(length=128, n_channels=2, rng=_rng(7))
    assert a == b


def test_gold_consistent_with_template_kinds():
    tpl = Categorization()
    for seed in range(30):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        if row.injections:
            assert row.injections[0].kind in ANOMALY_KINDS
