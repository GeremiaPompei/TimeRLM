from __future__ import annotations

import numpy as np

from anomalyxl.questions._time_buckets import QUARTER_LABELS_START, bucket_bounds
from anomalyxl.questions.start_time import StartTime


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def test_basic_options_shape():
    tpl = StartTime()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed))
        assert len(row.options) == 5
        assert len(set(row.options)) == 5
        assert row.answer in row.options
        assert "No Anomaly" in row.options


def test_no_anomaly_skips_injection():
    tpl = StartTime()
    for seed in range(200):
        row = tpl.build(length=256, n_channels=1, rng=_rng(seed))
        if row.answer == "No Anomaly":
            assert row.injections == []
            return
    raise AssertionError("No No-Anomaly gold seen")


def test_anomaly_gold_start_in_correct_bucket():
    tpl = StartTime()
    length = 1024
    for seed in range(60):
        row = tpl.build(length=length, n_channels=1, rng=_rng(seed))
        if row.answer == "No Anomaly":
            continue
        prefix = row.semantic_class
        bucket = QUARTER_LABELS_START.index(prefix)
        lo, hi = bucket_bounds(length, bucket)
        assert len(row.injections) == 1
        assert lo <= row.injections[0].t_start < hi


def test_semantic_class_matches_label_prefix():
    tpl = StartTime()
    for seed in range(30):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert tpl.semantic_class(row.answer) == row.semantic_class
