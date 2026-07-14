from __future__ import annotations

import numpy as np
import pytest

from anomalyxl.questions.identification import Identification


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def test_requires_at_least_three_channels():
    with pytest.raises(ValueError):
        Identification().build(length=128, n_channels=2, rng=_rng(0))


def test_basic_options_shape():
    tpl = Identification()
    for seed in range(20):
        row = tpl.build(length=256, n_channels=8, rng=_rng(seed))
        assert len(row.options) == 5
        assert len(set(row.options)) == 5
        assert row.answer in row.options
        assert "None" in row.options


def test_semantic_class_is_count():
    tpl = Identification()
    for seed in range(40):
        row = tpl.build(length=128, n_channels=8, rng=_rng(seed))
        size = 0 if row.answer == "None" else len(row.answer.split(", "))
        assert row.semantic_class == str(size)


def test_gold_injections_match_named_subset():
    tpl = Identification()
    for seed in range(60):
        row = tpl.build(length=128, n_channels=8, rng=_rng(seed))
        if row.answer == "None":
            named_subset = set()
        else:
            named_subset = set(row.answer.split(", "))
        # Channels in the named subset that are anomalous == named_subset itself.
        anomalous_named = {inj.channel for inj in row.injections} & named_subset
        assert anomalous_named == named_subset


def test_distractors_can_appear_outside_subset_at_high_channels():
    tpl = Identification()
    # With C=64 and per-channel p=0.2, distractors are nearly certain.
    found = False
    for seed in range(20):
        row = tpl.build(length=128, n_channels=64, rng=_rng(seed))
        named = set(row.answer.split(", ")) if row.answer != "None" else set()
        if any(inj.channel not in named for inj in row.injections):
            found = True
            break
    assert found
