from __future__ import annotations

import json

import numpy as np
import pytest

from anomalyxl.anomalies import ANOMALY_KINDS, MIN_WIDTH
from anomalyxl.questions.localize import Localize


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _parse(row_answer: str) -> dict:
    cat, _, payload = row_answer.partition(":")
    assert cat == "localize"
    return json.loads(payload)


def test_open_response_shape_and_gold_prefix():
    """No options block; gold is `localize:<json>` with `_L` always present."""
    tpl = Localize()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert row.options == []
        assert row.answer.startswith("localize:")
        payload = _parse(row.answer)
        assert payload["_L"] == 512
        assert "present" in payload


def test_both_present_and_absent_observed():
    """Uniform sampling must produce both branches over 50 seeds."""
    tpl = Localize()
    seen = set()
    for seed in range(50):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        seen.add(_parse(row.answer)["present"])
        if seen == {True, False}:
            return
    raise AssertionError(f"only observed: {seen}")


@pytest.mark.parametrize("length", [128, 512, 4096])
def test_present_gold_matches_injection(length: int):
    """When present, gold start/end must equal the injected window exactly."""
    tpl = Localize()
    found = False
    for seed in range(60):
        row = tpl.build(length=length, n_channels=1, rng=_rng(seed))
        payload = _parse(row.answer)
        if not payload["present"]:
            continue
        found = True
        assert len(row.injections) == 1
        spec = row.injections[0]
        assert payload["start"] == spec.t_start
        assert payload["end"] == spec.t_end
        assert spec.kind in ANOMALY_KINDS
        assert 0 <= spec.t_start < spec.t_end <= length
        assert spec.t_end - spec.t_start >= MIN_WIDTH
    assert found, f"no present rows in 60 seeds at L={length}"


def test_absent_means_no_injection():
    """When present=False, the row has no injections and no start/end keys."""
    tpl = Localize()
    found = False
    for seed in range(60):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        payload = _parse(row.answer)
        if payload["present"]:
            continue
        found = True
        assert row.injections == []
        assert "start" not in payload
        assert "end" not in payload
    assert found, "no absent rows in 60 seeds"


def test_question_contains_length():
    """The question template inlines L so the model knows the index range."""
    tpl = Localize()
    row = tpl.build(length=2048, n_channels=1, rng=_rng(0))
    assert "L = 2048" in row.question


def test_semantic_class_recoverable_from_gold():
    """`semantic_class(gold) == row.semantic_class` so analysis can re-bin."""
    tpl = Localize()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert tpl.semantic_class(row.answer) == row.semantic_class
        # Bin label encodes the L slice plus presence.
        assert row.semantic_class.endswith("_L512")
        assert row.semantic_class.startswith("localize_")


def test_no_coupling_for_tier_two():
    tpl = Localize()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert row.coupling is None


def test_registered_in_TEMPLATES():
    from anomalyxl.questions import TEMPLATES
    assert "localize" in TEMPLATES
    assert TEMPLATES["localize"] is Localize
