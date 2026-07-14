from __future__ import annotations

import numpy as np

from anomalyxl.questions.indicator import Indicator, _OPTIONS, _SEMANTIC


def _rng(s: int = 0) -> np.random.Generator:
    return np.random.default_rng(s)


def test_basic_options_shape():
    tpl = Indicator()
    for seed in range(10):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        assert len(row.options) == 5
        assert set(row.options) == set(_OPTIONS)
        assert row.answer in row.options


def test_leading_means_positive_lag():
    tpl = Indicator()
    for seed in range(400):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        if row.answer == _OPTIONS[0]:  # Leading
            assert row.coupling is not None
            assert row.coupling.kind == "coupled_lag_pos"
            assert row.coupling.lag > 0
            return
    raise AssertionError("Leading gold never observed")


def test_lagging_means_negative_lag():
    tpl = Indicator()
    for seed in range(400):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        if row.answer == _OPTIONS[1]:  # Lagging
            assert row.coupling is not None
            assert row.coupling.kind == "coupled_lag_neg"
            assert row.coupling.lag < 0
            return
    raise AssertionError("Lagging gold never observed")


def test_perfectly_correlated_zero_lag():
    tpl = Indicator()
    for seed in range(400):
        row = tpl.build(length=512, n_channels=2, rng=_rng(seed), paired_split=(0.5, 0.5))
        if row.answer == _OPTIONS[2]:  # Perfectly correlated
            assert row.coupling is not None and row.coupling.kind == "coupled_simul"
            assert row.coupling.lag == 0
            assert len(row.injections) == 2
            a_inj = next(i for i in row.injections if i.channel.startswith("A_"))
            b_inj = next(i for i in row.injections if i.channel.startswith("B_"))
            assert a_inj.t_start == b_inj.t_start
            return
    raise AssertionError("Perfectly correlated gold never observed")


def test_semantic_class_dispatch():
    tpl = Indicator()
    for opt, bin_name in _SEMANTIC.items():
        assert tpl.semantic_class(opt) == bin_name
