"""Env-side tests for `ts_datasets/anomalyxl_coarse.py`.

Covers the precise-localization scoring path: the answer-text parser
(`parse_precise_pred`), the gate that decides when `extra_metric_names`
fires, and `score_answer` dispatch on gold shape. Template-level tests
for the question builders live in `anomalyXL/tests/test_questions/`.
"""

from __future__ import annotations

import pytest

from ts_datasets.anomalyxl_coarse import (
    AnomalyXLCoarse,
    is_precise_gold,
    parse_precise_gold,
    parse_precise_pred,
)


# ----------------------------------------------------------------------------
# Parser: parse_precise_pred
# ----------------------------------------------------------------------------

PARSE_CASES = [
    # Bare integer.
    ("73214", 73214),
    # Common prose endings.
    ("Answer: 73214", 73214),
    ("The anomaly starts at index 73214", 73214),
    # Stray int before the answer — the LAST integer rule recovers the right one.
    ("0 < t_start < 73214", 73214),
    # Echoed gold "{idx}@{L}" — recognised via the gold regex, not the int scan.
    ("73214@128000", 73214),
    # Percentage commentary at the end — `(?!\s*%)` lookahead prevents the
    # parser from grabbing "57" instead of the index.
    ("index ~73214 (~57% in)", 73214),
    ("I see anomalies around 57% (start ~73214)", 73214),
    # JSON blob, value as string.
    ('{"answer": "73214"}', 73214),
    # JSON blob, value as number.
    ('{"answer": 73214}', 73214),
    # Empty / None input.
    ("", None),
]


@pytest.mark.parametrize("text,expected", PARSE_CASES)
def test_parse_precise_pred(text: str, expected: int | None):
    assert parse_precise_pred(text) == expected


def test_parse_precise_gold_roundtrip():
    assert parse_precise_gold("73214@128000") == (73214, 128000)
    assert is_precise_gold("73214@128000")
    assert not is_precise_gold("Level Shift")
    assert not is_precise_gold("")
    with pytest.raises(ValueError):
        parse_precise_gold("Level Shift")


# ----------------------------------------------------------------------------
# Metric gate: extra_metric_names only fires for precise-only category filters
# ----------------------------------------------------------------------------

# Fields that should appear when metrics are gated ON.
_PRECISE_METRIC_KEYS = {
    "acc_at_1pct",
    "acc_at_0p1pct",
    "acc_at_0p5pct",
    "acc_at_5pct",
    "mae_samples",
    "mae_fraction",
}


def test_metric_gate_off_when_no_filter():
    """No categories arg → mixed eval possible → no precise metrics emitted."""
    d = AnomalyXLCoarse()
    assert d.extra_metric_names == ()


def test_metric_gate_off_on_mixed_filter():
    d = AnomalyXLCoarse(categories=["precise_start_time", "categorization"])
    assert d.extra_metric_names == ()


def test_metric_gate_on_for_precise_only():
    d = AnomalyXLCoarse(categories=["precise_start_time", "precise_end_time"])
    assert set(d.extra_metric_names) == _PRECISE_METRIC_KEYS


def test_metric_gate_on_for_single_precise_category():
    d = AnomalyXLCoarse(categories=["precise_end_time"])
    assert set(d.extra_metric_names) == _PRECISE_METRIC_KEYS


def test_score_metrics_empty_when_gate_off():
    """Even when given a precise gold, the gated-off dataset returns ``{}``."""
    d = AnomalyXLCoarse()  # no filter → gate off
    assert d.score_metrics("100", "100@10000") == {}


# ----------------------------------------------------------------------------
# Scoring: score_answer dispatch + tolerance bands
# ----------------------------------------------------------------------------

def test_score_answer_precise_within_primary_band():
    """Primary band is `PRECISE_ACC_BANDS[0]` = 1%."""
    d = AnomalyXLCoarse()
    # 100 out of 10_000 → 1% off — exactly at the band edge, should pass.
    assert d.score_answer("200", "100@10000") == 1.0
    # 0% off — exact match.
    assert d.score_answer("100", "100@10000") == 1.0
    # 1.5% off — outside the primary band.
    assert d.score_answer("250", "100@10000") == 0.0


def test_score_answer_precise_with_unparseable_pred():
    d = AnomalyXLCoarse()
    assert d.score_answer("not a number", "100@10000") == 0.0


def test_score_answer_non_precise_falls_through_to_string_match():
    """Non-precise golds use the multiple-choice match path."""
    d = AnomalyXLCoarse()
    assert d.score_answer("Level Shift", "Level Shift") == 1.0
    assert d.score_answer("Transient Spike", "Level Shift") == 0.0


def test_score_metrics_bands_when_gate_on():
    d = AnomalyXLCoarse(categories=["precise_start_time"])
    # Exact hit — all bands satisfied, MAE zero.
    out = d.score_metrics("100", "100@10000")
    assert out["acc_at_1pct"] == 1.0
    assert out["acc_at_0p1pct"] == 1.0
    assert out["mae_samples"] == 0.0
    assert out["mae_fraction"] == 0.0

    # 1% off — 1pct and 5pct hit, tighter bands miss.
    out = d.score_metrics("200", "100@10000")
    assert out["acc_at_1pct"] == 1.0
    assert out["acc_at_5pct"] == 1.0
    assert out["acc_at_0p1pct"] == 0.0
    assert out["acc_at_0p5pct"] == 0.0
    assert out["mae_samples"] == 100.0
    assert out["mae_fraction"] == 0.01


def test_score_metrics_unparseable_pred_defaults_to_worst():
    """Failure to parse the prediction must not crash; bands miss, MAE = L."""
    d = AnomalyXLCoarse(categories=["precise_start_time"])
    out = d.score_metrics("garbage", "100@10000")
    assert all(out[k] == 0.0 for k in _PRECISE_METRIC_KEYS if k.startswith("acc_at_"))
    assert out["mae_samples"] == 10000.0
    assert out["mae_fraction"] == 1.0
