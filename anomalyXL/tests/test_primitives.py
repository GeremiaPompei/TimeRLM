from __future__ import annotations

import numpy as np
import pytest

from anomalyxl import primitives as P


L = 1024


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "fn",
    [P.white_noise, P.red_noise, P.ar1, P.sine, P.square, P.sawtooth,
     P.linear_drift, P.exp_drift, P.log_drift, P.constant],
)
def test_shape_and_dtype(fn):
    x = fn(L, _rng())
    assert x.shape == (L,)
    assert x.dtype == np.float64


@pytest.mark.parametrize(
    "fn",
    [P.white_noise, P.red_noise, P.ar1, P.sine, P.square, P.sawtooth,
     P.linear_drift, P.exp_drift, P.log_drift, P.constant],
)
def test_seed_reproducibility(fn):
    a = fn(L, _rng(123))
    b = fn(L, _rng(123))
    np.testing.assert_array_equal(a, b)


def test_seed_differs_across_seeds():
    a = P.white_noise(L, _rng(1))
    b = P.white_noise(L, _rng(2))
    assert not np.array_equal(a, b)


def test_white_noise_marginal_std_close_to_one():
    x = P.white_noise(16384, _rng(0), std=1.0)
    assert abs(x.std() - 1.0) < 0.05


def test_ar1_marginal_std_close_to_target():
    x = P.ar1(16384, _rng(0), phi=0.7, std=1.0)
    assert abs(x.std() - 1.0) < 0.1


def test_ar1_rejects_unstable_phi():
    with pytest.raises(ValueError):
        P.ar1(64, _rng(), phi=1.5)


def test_sine_period_matches_request():
    # Force a single specific period; check zero-crossing spacing.
    period = 64.0
    x = P.sine(4096, _rng(0), period_range=(period, period), amp_range=(1.0, 1.0))
    # Count zero crossings; ~ 2 per period.
    sign_changes = int(np.sum(np.diff(np.sign(x)) != 0))
    expected = 2 * 4096 / period
    assert abs(sign_changes - expected) / expected < 0.05


def test_square_has_two_levels():
    x = P.square(2048, _rng(0), period_range=(64.0, 64.0), amp_range=(1.0, 1.0))
    levels = np.unique(np.round(x, 6))
    assert set(levels) <= {-1.0, 0.0, 1.0}
    assert -1.0 in levels and 1.0 in levels


def test_sawtooth_monotone_ramps():
    period = 128.0
    x = P.sawtooth(1024, _rng(0), period_range=(period, period), amp_range=(1.0, 1.0))
    # Within a single period interior the sawtooth is strictly monotone.
    # Sample a window away from the wrap and check.
    window = x[10:60]
    diffs = np.diff(window)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_constant_is_constant():
    x = P.constant(512, _rng(0))
    assert x.std() == 0.0


def test_linear_drift_slope_is_constant():
    x = P.linear_drift(1024, _rng(0), slope_range=(1e-3, 1e-3))
    diffs = np.diff(x)
    assert np.allclose(diffs, 1e-3)


def test_draw_kind_dispatch():
    a = P.draw_kind("white_noise", L, _rng(7))
    b = P.white_noise(L, _rng(7))
    np.testing.assert_array_equal(a, b)


def test_draw_kind_unknown():
    with pytest.raises(KeyError):
        P.draw_kind("nope", L, _rng())
