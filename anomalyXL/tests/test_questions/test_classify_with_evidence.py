from __future__ import annotations

import json

import numpy as np
import pytest

from anomalyxl.anomalies import ANOMALY_KINDS
from anomalyxl.questions.categorization import _KIND_TO_LABEL
from anomalyxl.questions.classify_with_evidence import ClassifyWithEvidence


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _parse(row_answer: str) -> dict:
    cat, _, payload = row_answer.partition(":")
    assert cat == "classify_with_evidence"
    return json.loads(payload)


def test_gold_shape_and_keys():
    tpl = ClassifyWithEvidence()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert row.options == []
        payload = _parse(row.answer)
        assert set(payload) == {"kind", "start", "end", "_L"}
        assert payload["_L"] == 512


def test_gold_kind_matches_injection_label():
    tpl = ClassifyWithEvidence()
    for seed in range(40):
        row = tpl.build(length=1024, n_channels=2, rng=_rng(seed))
        assert len(row.injections) == 1
        spec = row.injections[0]
        payload = _parse(row.answer)
        assert spec.kind in ANOMALY_KINDS
        assert payload["kind"] == _KIND_TO_LABEL[spec.kind]
        assert payload["start"] == spec.t_start
        assert payload["end"] == spec.t_end


def test_question_lists_all_kind_labels():
    tpl = ClassifyWithEvidence()
    row = tpl.build(length=512, n_channels=1, rng=_rng(0))
    for label in _KIND_TO_LABEL.values():
        assert f'"{label}"' in row.question, f"label {label!r} missing from question"


def test_all_kinds_observed_over_seeds():
    """Uniform kind sampling — over 60 seeds, every kind should appear at least once."""
    tpl = ClassifyWithEvidence()
    kinds_seen = set()
    for seed in range(80):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        kinds_seen.add(_parse(row.answer)["kind"])
        if kinds_seen == set(_KIND_TO_LABEL.values()):
            return
    raise AssertionError(f"missing kinds: {set(_KIND_TO_LABEL.values()) - kinds_seen}")


def test_semantic_class_round_trip():
    tpl = ClassifyWithEvidence()
    for seed in range(20):
        row = tpl.build(length=512, n_channels=1, rng=_rng(seed))
        assert tpl.semantic_class(row.answer) == row.semantic_class
        assert row.semantic_class.endswith("_L512")
        assert row.semantic_class.startswith("classify_with_evidence_")


def test_question_contains_length():
    tpl = ClassifyWithEvidence()
    row = tpl.build(length=4096, n_channels=1, rng=_rng(0))
    assert "L = 4096" in row.question


def test_registered_in_TEMPLATES():
    from anomalyxl.questions import TEMPLATES
    assert "classify_with_evidence" in TEMPLATES
    assert TEMPLATES["classify_with_evidence"] is ClassifyWithEvidence
