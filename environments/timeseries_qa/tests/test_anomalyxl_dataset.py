"""Env-side tests for `ts_datasets/anomalyxl.py`.

Coverage:
- Strict JSON parser (`parse_structured_pred`): braces, escapes, missing
  keys, non-dict roots.
- Per-category scorers: zero on parse failure, correct on perfect
  predictions, expected partial credit on known mismatches.
- Set-matching (greedy IoU) for `localize_all_channels`.
- Composite reward for `lead_lag_with_magnitude`.

Template-level tests for the question builders live in
`anomalyXL/tests/test_questions/`.
"""

from __future__ import annotations

import json

import pytest

from ts_datasets.anomalyxl import (
    _score_classify_with_evidence,
    _score_lead_lag_with_magnitude,
    _score_localize,
    _score_localize_all_channels,
    _score_measure_magnitude,
    parse_structured_pred,
)


# --------------------------------------------------------------- JSON parser

def test_parse_simple():
    obj = parse_structured_pred('{"start": 5, "end": 10}', frozenset({"start", "end"}))
    assert obj == {"start": 5, "end": 10}


def test_parse_missing_required_key_returns_none():
    assert parse_structured_pred('{"start": 5}', frozenset({"start", "end"})) is None


def test_parse_handles_prose_around_json():
    obj = parse_structured_pred('I see it: {"start": 5, "end": 10}.', frozenset({"start"}))
    assert obj == {"start": 5, "end": 10}


def test_parse_handles_nested_objects():
    text = '{"anomalies": [{"channel": "ch_00", "start": 5, "end": 10}], "_L": 100}'
    obj = parse_structured_pred(text, frozenset({"anomalies"}))
    assert obj is not None
    assert len(obj["anomalies"]) == 1
    assert obj["anomalies"][0]["channel"] == "ch_00"


def test_parse_handles_string_with_brace():
    """A `}` inside a JSON string must not close the outer object."""
    text = '{"label": "weird} string", "value": 1}'
    obj = parse_structured_pred(text, frozenset({"label", "value"}))
    assert obj == {"label": "weird} string", "value": 1}


def test_parse_returns_none_on_empty():
    assert parse_structured_pred("", frozenset({"x"})) is None
    assert parse_structured_pred("no json here", frozenset({"x"})) is None


def test_parse_returns_none_on_malformed_json():
    assert parse_structured_pred('{"start": }', frozenset({"start"})) is None


def test_parse_returns_none_on_list_root():
    assert parse_structured_pred('[1, 2, 3]', frozenset()) is None


# --------------------------------------------------------------- localize

def test_localize_both_absent_perfect():
    gold = {"present": False, "_L": 1000}
    pred = {"present": False}
    m = _score_localize(pred, gold)
    assert m["iou_or_correct_empty"] == 1.0
    assert m["presence_correct"] == 1.0
    assert m["iou"] == 1.0


def test_localize_gold_absent_pred_present_zero():
    gold = {"present": False, "_L": 1000}
    pred = {"present": True, "start": 10, "end": 20}
    m = _score_localize(pred, gold)
    assert m["iou_or_correct_empty"] == 0.0
    assert m["presence_correct"] == 0.0


def test_localize_gold_present_pred_absent_zero():
    gold = {"present": True, "start": 10, "end": 20, "_L": 1000}
    pred = {"present": False}
    m = _score_localize(pred, gold)
    assert m["iou_or_correct_empty"] == 0.0
    assert m["presence_correct"] == 0.0


def test_localize_perfect_window_match():
    gold = {"present": True, "start": 10, "end": 20, "_L": 1000}
    pred = {"present": True, "start": 10, "end": 20}
    m = _score_localize(pred, gold)
    assert m["iou_or_correct_empty"] == 1.0
    assert m["iou"] == 1.0
    assert m["acc_at_iou_0p8"] == 1.0
    assert m["start_mae_fraction"] == 0.0
    assert m["end_mae_fraction"] == 0.0


def test_localize_partial_overlap_iou_known():
    # gold [10, 30), pred [20, 40) → intersection 10, union 30 → IoU = 1/3.
    gold = {"present": True, "start": 10, "end": 30, "_L": 1000}
    pred = {"present": True, "start": 20, "end": 40}
    m = _score_localize(pred, gold)
    assert abs(m["iou"] - 1/3) < 1e-9
    assert m["acc_at_iou_0p3"] == 1.0
    assert m["acc_at_iou_0p5"] == 0.0
    assert m["start_mae_fraction"] == 10 / 1000
    assert m["end_mae_fraction"] == 10 / 1000


def test_localize_parse_failure_zero():
    gold = {"present": True, "start": 10, "end": 20, "_L": 1000}
    m = _score_localize(None, gold)
    assert m["iou_or_correct_empty"] == 0.0
    assert m["presence_correct"] == 0.0


# --------------------------------------------------------------- classify_with_evidence

def test_classify_perfect():
    gold = {"kind": "Level Shift", "start": 10, "end": 20, "_L": 1000}
    pred = {"kind": "Level Shift", "start": 10, "end": 20}
    m = _score_classify_with_evidence(pred, gold)
    assert m["kind_x_iou"] == 1.0
    assert m["kind_correct"] == 1.0
    assert m["iou"] == 1.0
    assert m["kind_and_iou_above_0p5"] == 1.0


def test_classify_wrong_kind_right_window_zero_composite():
    gold = {"kind": "Level Shift", "start": 10, "end": 20, "_L": 1000}
    pred = {"kind": "Transient Spike", "start": 10, "end": 20}
    m = _score_classify_with_evidence(pred, gold)
    assert m["kind_correct"] == 0.0
    assert m["iou"] == 1.0
    assert m["kind_x_iou"] == 0.0


def test_classify_right_kind_wrong_window_iou_only():
    gold = {"kind": "Level Shift", "start": 10, "end": 30, "_L": 1000}
    pred = {"kind": "Level Shift", "start": 20, "end": 40}
    m = _score_classify_with_evidence(pred, gold)
    assert m["kind_correct"] == 1.0
    assert abs(m["iou"] - 1/3) < 1e-9
    assert abs(m["kind_x_iou"] - 1/3) < 1e-9
    assert m["kind_and_iou_above_0p5"] == 0.0


# --------------------------------------------------------------- measure_magnitude

def test_magnitude_perfect():
    gold = {"magnitude_sigma": 5.0, "_L": 1000}
    pred = {"magnitude_sigma": 5.0}
    m = _score_measure_magnitude(pred, gold)
    assert m["rel_err"] == 0.0
    assert m["acc_at_rel_10pct"] == 1.0
    assert m["acc_at_rel_25pct"] == 1.0


def test_magnitude_20pct_off_passes_25pct_band():
    gold = {"magnitude_sigma": 10.0, "_L": 1000}
    pred = {"magnitude_sigma": 12.0}  # 20% high
    m = _score_measure_magnitude(pred, gold)
    assert abs(m["rel_err"] - 0.20) < 1e-9
    assert m["acc_at_rel_10pct"] == 0.0
    assert m["acc_at_rel_25pct"] == 1.0
    assert m["acc_at_rel_50pct"] == 1.0


def test_magnitude_parse_failure_floors_rel_err_to_one():
    gold = {"magnitude_sigma": 5.0, "_L": 1000}
    m = _score_measure_magnitude(None, gold)
    assert m["rel_err"] == 1.0
    assert m["acc_at_rel_25pct"] == 0.0


# --------------------------------------------------------------- localize_all_channels

def test_lac_both_empty_perfect():
    gold = {"anomalies": [], "_L": 1000}
    pred = {"anomalies": []}
    m = _score_localize_all_channels(pred, gold)
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_lac_empty_gold_with_predictions_is_zero():
    gold = {"anomalies": [], "_L": 1000}
    pred = {"anomalies": [{"channel": "ch_00", "start": 10, "end": 20}]}
    m = _score_localize_all_channels(pred, gold)
    assert m["f1"] == 0.0
    assert m["fp"] == 1.0


def test_lac_two_of_three_recovered():
    gold = {
        "anomalies": [
            {"channel": "ch_00", "start": 10, "end": 20},
            {"channel": "ch_01", "start": 100, "end": 200},
            {"channel": "ch_02", "start": 500, "end": 600},
        ],
        "_L": 1000,
    }
    pred = {
        "anomalies": [
            {"channel": "ch_00", "start": 10, "end": 20},     # perfect match
            {"channel": "ch_01", "start": 100, "end": 200},   # perfect match
            # ch_02 missed
        ],
    }
    m = _score_localize_all_channels(pred, gold)
    assert m["tp"] == 2.0
    assert m["fp"] == 0.0
    assert m["fn"] == 1.0
    assert m["precision"] == 1.0
    assert abs(m["recall"] - 2/3) < 1e-9
    assert abs(m["f1"] - 2 * 1.0 * (2/3) / (1 + 2/3)) < 1e-9


def test_lac_wrong_channel_does_not_match():
    gold = {
        "anomalies": [{"channel": "ch_00", "start": 10, "end": 20}],
        "_L": 1000,
    }
    pred = {
        "anomalies": [{"channel": "ch_01", "start": 10, "end": 20}],
    }
    m = _score_localize_all_channels(pred, gold)
    assert m["tp"] == 0.0
    assert m["fp"] == 1.0
    assert m["fn"] == 1.0
    assert m["precision"] == 0.0


def test_lac_low_iou_rejected_below_threshold():
    """Match requires IoU > 0.3; a tiny overlap should not count."""
    gold = {"anomalies": [{"channel": "ch_00", "start": 0, "end": 100}], "_L": 1000}
    pred = {"anomalies": [{"channel": "ch_00", "start": 90, "end": 110}]}  # IoU = 10/110 ≈ 0.09
    m = _score_localize_all_channels(pred, gold)
    assert m["tp"] == 0.0


# --------------------------------------------------------------- lead_lag_with_magnitude

def test_lead_lag_perfect_lead():
    gold = {"direction": "lead", "lag_samples": 100, "_L": 10000}
    pred = {"direction": "lead", "lag_samples": 100}
    m = _score_lead_lag_with_magnitude(pred, gold)
    assert m["direction_correct"] == 1.0
    assert m["lag_mae_fraction"] == 0.0
    assert m["composite"] == 1.0
    assert m["lag_acc_at_1pct"] == 1.0
    assert m["lag_acc_at_5pct"] == 1.0


def test_lead_lag_wrong_direction_zero():
    gold = {"direction": "lead", "lag_samples": 100, "_L": 10000}
    pred = {"direction": "lag", "lag_samples": 100}
    m = _score_lead_lag_with_magnitude(pred, gold)
    assert m["direction_correct"] == 0.0
    assert m["composite"] == 0.0
    assert m["lag_mae_fraction"] == 1.0


def test_lead_lag_independent_match():
    gold = {"direction": "independent", "lag_samples": 0, "_L": 10000}
    pred = {"direction": "independent", "lag_samples": 9999}  # ignored
    m = _score_lead_lag_with_magnitude(pred, gold)
    assert m["direction_correct"] == 1.0
    assert m["composite"] == 1.0


def test_lead_lag_independent_mismatch():
    gold = {"direction": "independent", "lag_samples": 0, "_L": 10000}
    pred = {"direction": "lead", "lag_samples": 100}
    m = _score_lead_lag_with_magnitude(pred, gold)
    assert m["direction_correct"] == 0.0
    assert m["composite"] == 0.0
    assert m["lag_mae_fraction"] == 1.0


def test_lead_lag_composite_decays_with_lag_error():
    """Composite = max(0, 1 - min(1, lag_mae_frac / 0.05))."""
    L = 10000
    gold = {"direction": "lead", "lag_samples": 100, "_L": L}
    # 250 samples off out of L=10000 → 2.5% MAE; composite = 1 - 0.025/0.05 = 0.5.
    pred = {"direction": "lead", "lag_samples": 350}
    m = _score_lead_lag_with_magnitude(pred, gold)
    assert abs(m["composite"] - 0.5) < 1e-9
    # Lag MAE > 1% but < 5% → 1pct band 0, 5pct band 1.
    assert m["lag_acc_at_1pct"] == 0.0
    assert m["lag_acc_at_5pct"] == 1.0


# --------------------------------------------------------------- end-to-end via class

def test_extra_metric_names_union(smoke_shard):
    """The union of all 5 categories, over a generated smoke shard."""
    from ts_datasets.anomalyxl import AnomalyXL
    ds = AnomalyXL(repo_or_path=smoke_shard, eval_size=5)
    names = ds.extra_metric_names
    # All categories present → every per-category metric is in the union.
    assert "iou_or_correct_empty" in names
    assert "kind_x_iou" in names
    assert "acc_at_rel_25pct" in names
    assert "f1" in names
    assert "composite" in names


def test_score_answer_dispatch_via_class(smoke_shard):
    """`score_answer` correctly dispatches by gold prefix and applies the primary metric."""
    from ts_datasets.anomalyxl import AnomalyXL
    ds = AnomalyXL(repo_or_path=smoke_shard, eval_size=5)

    gold = 'localize:' + json.dumps({"present": True, "start": 10, "end": 20, "_L": 1000})
    perfect = '{"present": true, "start": 10, "end": 20}'
    assert ds.score_answer(perfect, gold) == 1.0
    assert ds.score_answer("no json here", gold) == 0.0


def test_score_metrics_projects_into_union(smoke_shard):
    """A measure_magnitude row's metric dict has all union keys, with non-mm ones at 0."""
    from ts_datasets.anomalyxl import AnomalyXL
    ds = AnomalyXL(repo_or_path=smoke_shard, eval_size=5)
    gold = 'measure_magnitude:' + json.dumps({"magnitude_sigma": 5.0, "_L": 1000})
    pred = '{"magnitude_sigma": 5.0}'
    m = ds.score_metrics(pred, gold)
    # Keys must be exactly the union, no more.
    assert set(m) == set(ds.extra_metric_names)
    assert m["acc_at_rel_25pct"] == 1.0
    # Irrelevant metrics defaulted to 0.
    assert m["f1"] == 0.0
    assert m["kind_x_iou"] == 0.0


# ------------------------------------------------- malformed answers score 0, never raise

@pytest.mark.parametrize("category,pred,gold,expected", [
    # A model that reports an anomaly but no usable window keeps presence credit,
    # scores 0 IoU, and must not raise.
    ("localize", '{"present": true, "start": null, "end": null}',
     {"present": True, "start": 10, "end": 20, "_L": 1000}, 0.0),
    ("localize", '{"present": true, "start": NaN, "end": 20}',
     {"present": True, "start": 10, "end": 20, "_L": 1000}, 0.0),
    ("localize", '{"present": true, "start": 1e400, "end": 20}',
     {"present": True, "start": 10, "end": 20, "_L": 1000}, 0.0),
    # No-anomaly stays first-class: a null window alongside present=false is still
    # a correct empty answer.
    ("localize", '{"present": false, "start": null, "end": null}',
     {"present": False, "_L": 1000}, 1.0),
    ("classify_with_evidence", '{"kind": "No anomaly", "start": null, "end": null}',
     {"kind": "No anomaly", "_L": 1000}, 1.0),
    ("classify_with_evidence", '{"kind": "Level Shift", "start": null, "end": null}',
     {"kind": "Level Shift", "start": 10, "end": 20, "_L": 1000}, 0.0),
    ("lead_lag_with_magnitude", '{"direction": "lead", "lag_samples": NaN}',
     {"direction": "lead", "lag_samples": 100, "_L": 1000}, 0.0),
    ("measure_magnitude", '{"magnitude_sigma": null}',
     {"magnitude_sigma": 5.0, "_L": 1000}, 0.0),
])
def test_malformed_values_score_zero_without_raising(smoke_shard, category, pred, gold, expected):
    from ts_datasets.anomalyxl import AnomalyXL
    ds = AnomalyXL(repo_or_path=smoke_shard, eval_size=5)
    assert ds.score_answer(pred, f"{category}:" + json.dumps(gold)) == expected
