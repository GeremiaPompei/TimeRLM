"""Indicator (Tier III paired): is A a leading/lagging indicator of B?

Five fixed options. "Leading" means A's anomaly precedes B's (positive lag);
"Lagging" means A's anomaly follows B's (negative lag). "No anomaly in one
or both" collapses `no_anomaly`, `one_sided_A`, and `one_sided_B`.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import make_coupling
from ._paired import build_paired_scenes, inject_for_coupling, merge_series
from .base import QuestionTemplate, Row

_OPTIONS: list[str] = [
    "Leading indicator (anomaly in A precedes anomaly in B)",
    "Lagging indicator (anomaly in A follows anomaly in B)",
    "Perfectly correlated (anomalies coincide in time)",
    "Not correlated (independent anomalies in A and B)",
    "No anomaly in one or both time-series",
]
_OPTION_TO_KIND_GROUP: dict[str, tuple[str, ...]] = {
    _OPTIONS[0]: ("coupled_lag_pos",),
    _OPTIONS[1]: ("coupled_lag_neg",),
    _OPTIONS[2]: ("coupled_simul",),
    _OPTIONS[3]: ("independent",),
    _OPTIONS[4]: ("one_sided_A", "one_sided_B", "no_anomaly"),
}
_SEMANTIC: dict[str, str] = {
    _OPTIONS[0]: "leading",
    _OPTIONS[1]: "lagging",
    _OPTIONS[2]: "simultaneous",
    _OPTIONS[3]: "independent",
    _OPTIONS[4]: "no_anomaly_or_one_sided",
}

_QUESTION = (
    "Two time-series are shown, with channels prefixed `A_` and `B_`. "
    "Is the anomaly in time-series 1 (A) a leading or lagging indicator of "
    "the anomaly in time-series 2 (B), or are the anomalies not correlated? "
    "By construction, coupled lead/lag offsets never exceed one quarter of "
    "the series length; independent (not correlated) anomalies are separated "
    "by more than that. Pick the option that best describes the relationship."
)


class Indicator(QuestionTemplate):
    category = "indicator"
    tier = 3
    min_channels = 2

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
    ) -> Row:
        scene_a, scene_b = build_paired_scenes(n_channels, length, rng, paired_split)

        gold_label = _OPTIONS[int(rng.integers(0, len(_OPTIONS)))]
        kind_group = _OPTION_TO_KIND_GROUP[gold_label]
        coupling_kind = kind_group[int(rng.integers(0, len(kind_group)))]
        coupling = make_coupling(length, rng, kind=coupling_kind)
        injections, coupling = inject_for_coupling(scene_a, scene_b, coupling, rng)

        options = list(_OPTIONS)
        rng.shuffle(options)

        return Row(
            series=merge_series([scene_a, scene_b]),
            question=_QUESTION,
            options=options,
            answer=gold_label,
            semantic_class=_SEMANTIC[gold_label],
            injections=injections,
            coupling=coupling,
        )

    def semantic_class(self, gold: str) -> str:
        return _SEMANTIC[gold]
