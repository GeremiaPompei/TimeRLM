from __future__ import annotations

import numpy as np

from anomalyxl.questions.presence import Presence


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def test_options_are_yes_no():
    tpl = Presence()
    for seed in range(20):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        assert sorted(row.options) == ["No", "Yes"]
        assert row.answer in row.options


def test_yes_gold_has_injection_no_gold_has_none():
    tpl = Presence()
    seen_yes = seen_no = False
    for seed in range(200):
        row = tpl.build(length=128, n_channels=1, rng=_rng(seed))
        if row.answer == "Yes":
            assert len(row.injections) == 1
            seen_yes = True
        else:
            assert row.injections == []
            seen_no = True
        if seen_yes and seen_no:
            return
    raise AssertionError("Did not observe both Yes and No across 200 seeds")


def test_semantic_class_matches_gold():
    tpl = Presence()
    for seed in range(20):
        row = tpl.build(length=128, n_channels=2, rng=_rng(seed))
        assert row.semantic_class == row.answer
