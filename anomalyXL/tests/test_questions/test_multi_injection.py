from __future__ import annotations

import numpy as np
import pytest

from anomalyxl.anomalies import ANOMALY_KINDS, MIN_WIDTH, MAX_WIDTH
from anomalyxl.questions._multi_injection import place_k_anomalies
from anomalyxl.scene import build_scene


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def _fresh_scene(n_channels: int = 16, length: int = 4096, seed: int = 0):
    return build_scene(n_channels, length, _rng(seed))


def test_k_zero_returns_empty_and_does_not_touch_scene():
    """K=0 is a first-class case — no injections, scene unchanged."""
    scene = _fresh_scene()
    snapshot = {name: arr.copy() for name, arr in scene.channels.items()}
    placed = place_k_anomalies(scene, 0, _rng(1))
    assert placed == []
    for name, before in snapshot.items():
        np.testing.assert_array_equal(scene.channels[name], before)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_k_anomalies_on_distinct_channels(k: int):
    """For 1<=K<=3, every placed injection lives on a distinct channel."""
    for seed in range(20):
        scene = _fresh_scene(seed=seed)
        placed = place_k_anomalies(scene, k, _rng(seed + 100))
        assert len(placed) == k
        channels = [s.channel for s in placed]
        assert len(set(channels)) == k, f"expected K distinct channels, got {channels}"
        # All chosen channels must exist in the scene.
        for ch in channels:
            assert ch in scene.channels


def test_windows_stay_in_bounds():
    """Every placed window satisfies 0 <= t_start < t_end <= L with min-width >= MIN_WIDTH
    and width <= MAX_WIDTH (absolute, plan-locked)."""
    L = 8192
    for seed in range(20):
        scene = build_scene(16, L, _rng(seed))
        placed = place_k_anomalies(scene, 3, _rng(seed + 100))
        for spec in placed:
            assert 0 <= spec.t_start < spec.t_end <= L
            width = spec.t_end - spec.t_start
            assert MIN_WIDTH <= width <= MAX_WIDTH


def test_injection_kinds_in_known_set():
    scene = _fresh_scene()
    placed = place_k_anomalies(scene, 3, _rng(7))
    for spec in placed:
        assert spec.kind in ANOMALY_KINDS


def test_distinct_channel_enforced_when_k_exceeds_pool():
    """Asking for more anomalies than channels with distinct-channel constraint
    must raise — silently double-stacking would corrupt the K=3 invariant."""
    scene = build_scene(2, 1024, _rng(0))
    with pytest.raises(ValueError, match="distinct-channel"):
        place_k_anomalies(scene, 3, _rng(0))


def test_scene_actually_mutated():
    """Injection must alter the chosen channels (not just record specs)."""
    scene = _fresh_scene()
    snapshot = {name: arr.copy() for name, arr in scene.channels.items()}
    placed = place_k_anomalies(scene, 2, _rng(11))
    touched = {s.channel for s in placed}
    untouched = set(scene.channels) - touched
    for name in touched:
        diff = scene.channels[name] - snapshot[name]
        assert np.any(diff != 0), f"channel {name} should have been mutated"
    for name in untouched:
        np.testing.assert_array_equal(scene.channels[name], snapshot[name])


def test_negative_k_rejected():
    scene = _fresh_scene()
    with pytest.raises(ValueError, match="non-negative"):
        place_k_anomalies(scene, -1, _rng(0))


def test_deterministic_under_same_seed():
    """Same scene + same rng seed → identical placement."""
    s1 = _fresh_scene(seed=42)
    s2 = _fresh_scene(seed=42)
    p1 = place_k_anomalies(s1, 3, _rng(99))
    p2 = place_k_anomalies(s2, 3, _rng(99))
    assert [(s.channel, s.t_start, s.t_end, s.kind) for s in p1] == \
           [(s.channel, s.t_start, s.t_end, s.kind) for s in p2]


def test_same_channel_allowed_rejects_overlap():
    """When `same_channel_allowed=True`, repeated placements on a single-channel
    scene must produce non-overlapping windows (or raise on saturation)."""
    L = 2048
    scene = build_scene(1, L, _rng(0))
    placed = place_k_anomalies(
        scene, 2, _rng(5), same_channel_allowed=True, max_attempts=64,
    )
    assert len(placed) == 2
    a, b = sorted(placed, key=lambda s: s.t_start)
    assert a.t_end <= b.t_start or b.t_end <= a.t_start
    assert a.channel == b.channel  # single-channel scene
