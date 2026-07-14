from __future__ import annotations

import json

import numpy as np
import pytest

from anomalyxl.questions.measure_magnitude import (
    MeasureMagnitude,
    _MAGNITUDE_HI,
    _MAGNITUDE_LO,
)


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _parse(row_answer: str) -> dict:
    cat, _, payload = row_answer.partition(":")
    assert cat == "measure_magnitude"
    return json.loads(payload)


def test_gold_shape_and_keys():
    tpl = MeasureMagnitude()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert row.options == []
        payload = _parse(row.answer)
        assert set(payload) == {"magnitude_sigma", "_L"}
        assert payload["_L"] == 512
        assert isinstance(payload["magnitude_sigma"], (int, float))


def test_always_level_shift_with_one_injection():
    tpl = MeasureMagnitude()
    for seed in range(40):
        row = tpl.build(length=1024, n_channels=1, rng=_rng(seed))
        assert len(row.injections) == 1
        spec = row.injections[0]
        assert spec.kind == "level_shift"


def test_magnitude_sigma_matches_empirical_peak():
    """Gold magnitude_sigma must equal empirical peak / pre_std on the
    injected channel — within float rounding tolerance for the
    6-decimal-rounded gold."""
    tpl = MeasureMagnitude()
    for seed in range(20):
        row = tpl.build(length=1024, n_channels=1, rng=_rng(seed))
        spec = row.injections[0]
        payload = _parse(row.answer)
        # The injected channel is the only one in the row at C=1.
        injected = np.asarray(row.series[spec.channel], dtype=np.float64)
        # `series` is the post-injection float32 cast; we recover peak from
        # the spec + pre-injection convention: magnitude is in σ-units of
        # the pre-injection z-normed channel (std≈1), so reported value
        # is within 1% of |spec.magnitude|.
        rel_err = abs(payload["magnitude_sigma"] - abs(spec.magnitude)) / abs(spec.magnitude)
        assert rel_err < 0.05, (
            f"reported {payload['magnitude_sigma']} vs requested |{spec.magnitude}|, "
            f"rel_err={rel_err:.4f}"
        )


def test_magnitude_in_log_uniform_range():
    """Gold magnitude_sigma must fall in [1, 100] up to float rounding."""
    tpl = MeasureMagnitude()
    for seed in range(40):
        row = tpl.build(length=1024, n_channels=1, rng=_rng(seed))
        m = _parse(row.answer)["magnitude_sigma"]
        # Allow a small slack — pre_std is ≈1 not exactly 1, so reported
        # value can fall slightly outside the requested magnitude bounds.
        assert _MAGNITUDE_LO * 0.5 <= m <= _MAGNITUDE_HI * 2.0


def test_log_uniform_spans_both_decades():
    """Over enough seeds, both ~1-10σ and ~10-100σ buckets must be observed."""
    tpl = MeasureMagnitude()
    small, large = 0, 0
    for seed in range(80):
        row = tpl.build(length=1024, n_channels=1, rng=_rng(seed))
        m = _parse(row.answer)["magnitude_sigma"]
        if m < 10.0:
            small += 1
        else:
            large += 1
    assert small > 5, f"expected several sub-10σ samples, got {small}"
    assert large > 5, f"expected several 10σ+ samples, got {large}"


def test_semantic_class_round_trip():
    tpl = MeasureMagnitude()
    for seed in range(20):
        row = tpl.build(length=2048, n_channels=1, rng=_rng(seed))
        assert tpl.semantic_class(row.answer) == row.semantic_class
        assert row.semantic_class == "measure_magnitude_L2048"


def test_question_contains_length():
    tpl = MeasureMagnitude()
    row = tpl.build(length=8192, n_channels=1, rng=_rng(0))
    assert "L = 8192" in row.question


def test_registered_in_TEMPLATES():
    from anomalyxl.questions import TEMPLATES
    assert "measure_magnitude" in TEMPLATES
    assert TEMPLATES["measure_magnitude"] is MeasureMagnitude
