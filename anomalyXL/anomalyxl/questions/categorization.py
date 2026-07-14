"""Categorization (Tier II): which anomaly kind appears in the series.

Gold is sampled from `ANOMALY_KINDS ∪ {"no_anomaly"}`; the scene then has the
matching injection built (or no injection at all for "No Anomaly"). Options
are always `4 distinct anomaly kinds + "No Anomaly"` — the random drop varies
the distractor set across rows.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..scene import build_scene
from .base import QuestionTemplate, Row

# ARFBench option labels for the anomaly kinds (Appendix C.2 wording).
_KIND_TO_LABEL: dict[str, str] = {
    "level_shift": "Level Shift",
    "transient_spike": "Transient Spike",
    "change_in_seasonality": "Change in Seasonality",
    "change_in_variance": "Change in Variance",
    "change_in_trend": "Change in Trend",
}
_NO_ANOMALY_LABEL = "No Anomaly"

# ARFBench Appendix C.2: Categorization collapses Seasonality + Variance.
_SEMANTIC_BIN: dict[str, str] = {
    "Level Shift": "Level Shift",
    "Transient Spike": "Transient Spike",
    "Change in Seasonality": "Change in Seasonality or Variance",
    "Change in Variance": "Change in Seasonality or Variance",
    "Change in Trend": "Change in Trend",
    "No Anomaly": "No Anomaly",
}

_QUESTION = (
    "Which category best describes the anomaly in the time series? "
    "If no anomaly is present, choose 'No Anomaly'."
)


class Categorization(QuestionTemplate):
    category = "categorization"
    tier = 2

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
    ) -> Row:
        del paired_split
        if n_channels < 1:
            raise ValueError(f"Categorization needs >=1 channel, got {n_channels}")

        scene = build_scene(n_channels, length, rng)

        # Uniform pick over {5 kinds, no_anomaly}.
        choices: tuple[str, ...] = ANOMALY_KINDS + ("no_anomaly",)
        gold_kind = choices[int(rng.integers(0, len(choices)))]
        gold_label = _NO_ANOMALY_LABEL if gold_kind == "no_anomaly" else _KIND_TO_LABEL[gold_kind]

        injections = []
        if gold_kind != "no_anomaly":
            target = scene.names[int(rng.integers(0, n_channels))]
            spec = make_injection(length, rng, kind=gold_kind, channel=target)
            scene.channels[target] = inject(scene.channels[target], spec, rng)
            injections.append(spec)

        options = _build_options(gold_label, rng)

        series = {name: arr.astype(np.float32).tolist() for name, arr in scene.channels.items()}

        return Row(
            series=series,
            question=_QUESTION,
            options=options,
            answer=gold_label,
            semantic_class=_SEMANTIC_BIN[gold_label],
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        return _SEMANTIC_BIN[gold]


def _build_options(gold_label: str, rng: np.random.Generator) -> list[str]:
    """Always 5 options: 4 distinct anomaly-kind labels + 'No Anomaly'.

    If the gold is an anomaly kind, drop one *other* anomaly kind so the gold
    survives; if the gold is 'No Anomaly', drop any one anomaly kind. Shuffle
    deterministically off the supplied rng.
    """
    all_anomaly_labels = list(_KIND_TO_LABEL.values())
    if gold_label == _NO_ANOMALY_LABEL:
        droppable = all_anomaly_labels
    else:
        droppable = [lab for lab in all_anomaly_labels if lab != gold_label]
    drop = droppable[int(rng.integers(0, len(droppable)))]
    kept = [lab for lab in all_anomaly_labels if lab != drop]
    options = kept + [_NO_ANOMALY_LABEL]
    rng.shuffle(options)
    return options
