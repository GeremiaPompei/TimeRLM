from __future__ import annotations

import json

import numpy as np
import pytest

from anomalyxl.questions.lead_lag_with_magnitude import (
    LeadLagWithMagnitude,
    _DIRECTIONS,
)


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _parse(row_answer: str) -> dict:
    cat, _, payload = row_answer.partition(":")
    assert cat == "lead_lag_with_magnitude"
    return json.loads(payload)


def test_gold_shape_and_keys():
    tpl = LeadLagWithMagnitude()
    for seed in range(20):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        assert row.options == []
        payload = _parse(row.answer)
        assert set(payload) == {"direction", "lag_samples", "_L"}
        assert payload["direction"] in _DIRECTIONS
        assert payload["lag_samples"] >= 0
        assert payload["_L"] == 2048


def test_all_three_directions_observed():
    tpl = LeadLagWithMagnitude()
    seen = set()
    for seed in range(60):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        seen.add(_parse(row.answer)["direction"])
        if seen == set(_DIRECTIONS):
            return
    raise AssertionError(f"missing directions: {set(_DIRECTIONS) - seen}")


def test_lead_means_A_precedes_B():
    """For direction=lead, A's anomaly t_start < B's anomaly t_start, and the
    delta equals gold lag_samples."""
    tpl = LeadLagWithMagnitude()
    for seed in range(400):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        payload = _parse(row.answer)
        if payload["direction"] != "lead":
            continue
        a_inj = next(i for i in row.injections if i.channel.startswith("A_"))
        b_inj = next(i for i in row.injections if i.channel.startswith("B_"))
        assert a_inj.t_start < b_inj.t_start
        assert b_inj.t_start - a_inj.t_start == payload["lag_samples"]
        assert row.coupling is not None and row.coupling.kind == "coupled_lag_pos"
        return
    raise AssertionError("never observed direction=lead")


def test_lag_means_A_follows_B():
    tpl = LeadLagWithMagnitude()
    for seed in range(400):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        payload = _parse(row.answer)
        if payload["direction"] != "lag":
            continue
        a_inj = next(i for i in row.injections if i.channel.startswith("A_"))
        b_inj = next(i for i in row.injections if i.channel.startswith("B_"))
        assert a_inj.t_start > b_inj.t_start
        assert a_inj.t_start - b_inj.t_start == payload["lag_samples"]
        assert row.coupling is not None and row.coupling.kind == "coupled_lag_neg"
        return
    raise AssertionError("never observed direction=lag")


def test_independent_has_two_anomalies_disjoint_windows():
    tpl = LeadLagWithMagnitude()
    for seed in range(400):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        payload = _parse(row.answer)
        if payload["direction"] != "independent":
            continue
        assert len(row.injections) == 2
        a_inj = next(i for i in row.injections if i.channel.startswith("A_"))
        b_inj = next(i for i in row.injections if i.channel.startswith("B_"))
        # Disjoint windows: one ends before the other starts.
        assert a_inj.t_end <= b_inj.t_start or b_inj.t_end <= a_inj.t_start
        assert row.coupling is not None and row.coupling.kind == "independent"
        return
    raise AssertionError("never observed direction=independent")


def test_lag_samples_within_bounds():
    """For lead/lag, lag_samples ∈ [1, L/4]."""
    tpl = LeadLagWithMagnitude()
    L = 4096
    for seed in range(60):
        row = tpl.build(length=L, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        payload = _parse(row.answer)
        if payload["direction"] == "independent":
            continue
        assert 1 <= payload["lag_samples"] <= L // 4


def test_independent_lag_is_zero_in_gold():
    """For independent direction, gold lag_samples must be 0 (ignored downstream)."""
    tpl = LeadLagWithMagnitude()
    for seed in range(80):
        row = tpl.build(length=2048, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        payload = _parse(row.answer)
        if payload["direction"] == "independent":
            assert payload["lag_samples"] == 0


def test_min_channels_enforced():
    tpl = LeadLagWithMagnitude()
    assert tpl.min_channels == 2
    with pytest.raises(ValueError, match=">=2"):
        tpl.build(length=2048, n_channels=1, rng=_rng(0))


def test_semantic_class_round_trip():
    tpl = LeadLagWithMagnitude()
    for seed in range(20):
        row = tpl.build(length=1024, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        assert tpl.semantic_class(row.answer) == row.semantic_class
        assert row.semantic_class.endswith("_L1024")


def test_registered_in_TEMPLATES():
    from anomalyxl.questions import TEMPLATES
    assert "lead_lag_with_magnitude" in TEMPLATES
    assert TEMPLATES["lead_lag_with_magnitude"] is LeadLagWithMagnitude
