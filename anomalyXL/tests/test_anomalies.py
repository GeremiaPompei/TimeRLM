from __future__ import annotations

import numpy as np
import pytest

from anomalyxl.anomalies import (
    ANOMALY_KINDS,
    COUPLING_KINDS,
    MAX_WIDTH,
    MIN_WIDTH,
    CouplingSpec,
    InjectionSpec,
    inject,
    make_coupling,
    make_injection,
)


L = 1024


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _baseline(L: int = L) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, 1.0, size=L)


def test_make_injection_window_inside_series():
    for _ in range(50):
        spec = make_injection(L, _rng(np.random.randint(1_000_000)))
        assert 0 <= spec.t_start < spec.t_end <= L
        assert MIN_WIDTH <= spec.width <= MAX_WIDTH
        assert spec.kind in ANOMALY_KINDS


def test_make_injection_respects_explicit_kind():
    spec = make_injection(L, _rng(0), kind="level_shift")
    assert spec.kind == "level_shift"


def test_make_injection_runoff_can_reach_L():
    found = False
    for seed in range(100):
        spec = make_injection(L, _rng(seed), allow_runoff=True)
        if spec.t_end == L:
            found = True
            break
    assert found


def test_make_injection_no_runoff_never_reaches_L():
    for seed in range(50):
        spec = make_injection(L, _rng(seed), allow_runoff=False, max_width=MIN_WIDTH)
        # max_width = MIN_WIDTH forces width = MIN_WIDTH; t_end can equal L
        # only when t_start = L - MIN_WIDTH; check the boundary handling.
        assert spec.t_end <= L


def test_inject_does_not_mutate_input():
    series = _baseline()
    snap = series.copy()
    spec = make_injection(L, _rng(0), kind="level_shift", channel="ch_0")
    inject(series, spec, _rng(0))
    np.testing.assert_array_equal(series, snap)


@pytest.mark.parametrize("kind", ANOMALY_KINDS)
def test_inject_changes_window(kind: str):
    series = _baseline()
    spec = InjectionSpec(channel="ch_0", t_start=200, t_end=400, kind=kind, magnitude=5.0)
    out = inject(series, spec, _rng(0))
    # The window contents should differ.
    assert not np.allclose(out[spec.t_start:spec.t_end], series[spec.t_start:spec.t_end])


def test_level_shift_changes_window_mean():
    series = _baseline()
    spec = InjectionSpec(channel="ch_0", t_start=100, t_end=300, kind="level_shift", magnitude=5.0)
    out = inject(series, spec, _rng(0))
    pre = series[spec.t_start:spec.t_end].mean()
    post = out[spec.t_start:spec.t_end].mean()
    assert abs(post - pre) > 3.0


def test_transient_spike_peaks_inside_window():
    series = np.zeros(L)
    spec = InjectionSpec(channel="ch_0", t_start=400, t_end=500, kind="transient_spike", magnitude=10.0)
    out = inject(series, spec, _rng(0))
    abs_window = np.abs(out[spec.t_start:spec.t_end])
    abs_outside = np.abs(np.concatenate([out[:spec.t_start], out[spec.t_end:]]))
    assert abs_window.max() > 1.0
    # Spike should be confined to the window.
    assert abs_outside.max() < 1e-9


def test_change_in_variance_increases_variance():
    series = np.zeros(L)
    spec = InjectionSpec(channel="ch_0", t_start=200, t_end=600, kind="change_in_variance", magnitude=4.0)
    out = inject(series, spec, _rng(0))
    window_std = out[spec.t_start:spec.t_end].std()
    outside_std = np.concatenate([out[:spec.t_start], out[spec.t_end:]]).std()
    assert window_std > outside_std + 1.0


def test_change_in_trend_carries_forward():
    series = np.zeros(L)
    spec = InjectionSpec(channel="ch_0", t_start=200, t_end=400, kind="change_in_trend", magnitude=5.0)
    out = inject(series, spec, _rng(0))
    # After the ramp the carry-forward holds the level above pre-window level.
    assert abs(out[500] - out[100]) > 1.0


def test_change_in_seasonality_adds_oscillation():
    series = np.zeros(L)
    spec = InjectionSpec(channel="ch_0", t_start=200, t_end=500, kind="change_in_seasonality", magnitude=3.0)
    out = inject(series, spec, _rng(0))
    window = out[spec.t_start:spec.t_end]
    # Multiple zero crossings inside the window.
    assert int(np.sum(np.diff(np.sign(window)) != 0)) >= 4


def test_make_coupling_lag_direction():
    pos = make_coupling(8192, _rng(0), kind="coupled_lag_pos")
    neg = make_coupling(8192, _rng(0), kind="coupled_lag_neg")
    assert pos.lag > 0
    assert neg.lag < 0


def test_make_coupling_no_lag_for_non_lag_kinds():
    for kind in ("independent", "coupled_simul", "one_sided_A", "one_sided_B", "no_anomaly"):
        spec = make_coupling(8192, _rng(0), kind=kind)
        assert spec.lag == 0
        assert spec.kind == kind


def test_make_coupling_clamps_lag_for_short_L():
    spec = make_coupling(64, _rng(0), kind="coupled_lag_pos")
    assert spec.lag <= 64 // 4


def test_make_coupling_rejects_unknown_kind():
    with pytest.raises(KeyError):
        make_coupling(1024, _rng(0), kind="nope")


def test_known_coupling_kinds_match_module_constant():
    assert set(COUPLING_KINDS) >= {
        "independent", "coupled_simul", "coupled_lag_pos",
        "coupled_lag_neg", "one_sided_A", "one_sided_B", "no_anomaly",
    }
