from __future__ import annotations

import json

import numpy as np
import pytest

from anomalyxl.anomalies import MAX_WIDTH, MIN_WIDTH
from anomalyxl.questions.localize_all_channels import LocalizeAllChannels


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _parse(row_answer: str) -> dict:
    cat, _, payload = row_answer.partition(":")
    assert cat == "localize_all_channels"
    return json.loads(payload)


def test_min_channels_enforced():
    tpl = LocalizeAllChannels()
    assert tpl.min_channels == 16
    with pytest.raises(ValueError, match=">=16"):
        tpl.build(length=1024, n_channels=8, rng=_rng(0))


def test_gold_shape_and_keys():
    tpl = LocalizeAllChannels()
    for seed in range(20):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        assert row.options == []
        payload = _parse(row.answer)
        assert set(payload) == {"anomalies", "_L"}
        assert payload["_L"] == 4096
        assert isinstance(payload["anomalies"], list)


def test_K_distribution_covers_full_range():
    """Over enough seeds, K∈{0,1,2,3} should all appear."""
    tpl = LocalizeAllChannels()
    seen_K = set()
    for seed in range(120):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        k = len(_parse(row.answer)["anomalies"])
        seen_K.add(k)
        if seen_K == {0, 1, 2, 3}:
            return
    missing = {0, 1, 2, 3} - seen_K
    raise AssertionError(f"missing K values: {missing}")


def test_K_matches_injection_count():
    tpl = LocalizeAllChannels()
    for seed in range(40):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        payload = _parse(row.answer)
        assert len(payload["anomalies"]) == len(row.injections)


def test_all_events_distinct_channels():
    """Per the _multi_injection contract — every event lives on a distinct channel."""
    tpl = LocalizeAllChannels()
    for seed in range(60):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        channels = [e["channel"] for e in events]
        assert len(set(channels)) == len(channels)


def test_event_windows_in_bounds():
    tpl = LocalizeAllChannels()
    L = 4096
    for seed in range(40):
        row = tpl.build(length=L, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        for ev in events:
            assert 0 <= ev["start"] < ev["end"] <= L
            assert MIN_WIDTH <= ev["end"] - ev["start"] <= MAX_WIDTH


def test_event_channel_in_series():
    tpl = LocalizeAllChannels()
    for seed in range(40):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        for ev in events:
            assert ev["channel"] in row.series


def test_event_matches_injection():
    """Each gold event must align with one InjectionSpec."""
    tpl = LocalizeAllChannels()
    for seed in range(40):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        inj_by_channel = {
            inj.channel: (inj.t_start, inj.t_end) for inj in row.injections
        }
        for ev in events:
            assert ev["channel"] in inj_by_channel
            assert (ev["start"], ev["end"]) == inj_by_channel[ev["channel"]]


def test_events_sorted_for_stable_gold():
    """Gold events are sorted by (channel, start) for hashability of the JSON gold."""
    tpl = LocalizeAllChannels()
    for seed in range(20):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        sorted_events = sorted(events, key=lambda e: (e["channel"], e["start"]))
        assert events == sorted_events


def test_K_zero_branch_empty_payload():
    """Find at least one seed producing K=0; verify its payload is the empty list
    AND row.injections is empty."""
    tpl = LocalizeAllChannels()
    for seed in range(120):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        events = _parse(row.answer)["anomalies"]
        if events == []:
            assert row.injections == []
            return
    raise AssertionError("never observed K=0 in 120 seeds")


def test_semantic_class_round_trip():
    tpl = LocalizeAllChannels()
    for seed in range(30):
        row = tpl.build(length=4096, n_channels=16, rng=_rng(seed))
        assert tpl.semantic_class(row.answer) == row.semantic_class
        assert row.semantic_class.startswith("localize_all_channels_K")
        assert row.semantic_class.endswith("_L4096")


def test_question_contains_length_and_C():
    tpl = LocalizeAllChannels()
    row = tpl.build(length=8192, n_channels=16, rng=_rng(0))
    assert "L = 8192" in row.question
    assert "C = 16" in row.question


def test_registered_in_TEMPLATES():
    from anomalyxl.questions import TEMPLATES
    assert "localize_all_channels" in TEMPLATES
    assert TEMPLATES["localize_all_channels"] is LocalizeAllChannels
