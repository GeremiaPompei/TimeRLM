from __future__ import annotations

import numpy as np
import pytest

from anomalyxl.scene import build_scene


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_basic_shape():
    scene = build_scene(n_channels=4, length=512, rng=_rng(0))
    assert len(scene.channels) == 4
    for name, arr in scene.channels.items():
        assert arr.shape == (512,)
        assert arr.dtype == np.float64
        assert name.startswith("ch_")


def test_channel_naming_padding():
    s = build_scene(n_channels=16, length=64, rng=_rng(0))
    names = list(s.channels.keys())
    assert names[0] == "ch_00"
    assert names[-1] == "ch_15"


def test_name_prefix_for_paired():
    s_a = build_scene(n_channels=3, length=64, rng=_rng(0), name_prefix="A_")
    s_b = build_scene(n_channels=3, length=64, rng=_rng(1), name_prefix="B_")
    assert all(n.startswith("A_ch_") for n in s_a.channels)
    assert all(n.startswith("B_ch_") for n in s_b.channels)
    assert set(s_a.channels.keys()).isdisjoint(s_b.channels.keys())


def test_unit_normalization():
    scene = build_scene(n_channels=8, length=4096, rng=_rng(0))
    for arr in scene.channels.values():
        assert abs(arr.mean()) < 1e-6
        # After normalization std should be ~1 (constant channels collapse to 0).
        assert abs(arr.std() - 1.0) < 1e-6 or arr.std() == 0.0


def test_seed_reproducibility():
    a = build_scene(n_channels=4, length=256, rng=_rng(42))
    b = build_scene(n_channels=4, length=256, rng=_rng(42))
    for name in a.channels:
        np.testing.assert_array_equal(a.channels[name], b.channels[name])
    assert a.base_components == b.base_components


def test_base_components_recorded():
    scene = build_scene(n_channels=4, length=512, rng=_rng(0))
    for name, components in scene.base_components.items():
        assert len(components) >= 1
        assert components[0] in {"white_noise", "red_noise", "ar1"}


def test_zero_channels_is_empty():
    s = build_scene(n_channels=0, length=100, rng=_rng(0))
    assert s.channels == {}
    assert s.length == 0
    assert s.names == []


def test_periodic_p_zero_yields_no_periodic_component():
    s = build_scene(n_channels=8, length=128, rng=_rng(0), periodic_p=0.0, trend_p=0.0)
    for comps in s.base_components.values():
        assert len(comps) == 1


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        build_scene(n_channels=2, length=0, rng=_rng())
