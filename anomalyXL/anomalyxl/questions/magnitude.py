"""Magnitude (Tier II): how large is the anomaly's peak deviation, in units
of `channel_std`?

Bins are hardcoded log-scale edges (see `PLAN.md §13.1` for the open-item
trade-off vs. ARFBench's LLM-chosen numerics). Gold is picked first; the
sampled magnitude lives strictly inside the chosen bin. The injection is
always a `level_shift` so the peak deviation equals `magnitude * channel_std`
exactly — kinds whose peak doesn't equal the requested magnitude would
muddy the gold.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import InjectionSpec, MIN_WIDTH, MAX_WIDTH, inject
from ..scene import build_scene
from .base import QuestionTemplate, Row

# Closed-open bins on `magnitude` in units of channel_std.
_BINS: tuple[tuple[str, float, float], ...] = (
    ("Smallest", 1.0, 3.0),
    ("Small", 3.0, 10.0),
    ("Medium", 10.0, 30.0),
    ("Large", 30.0, 100.0),
)
_NO_ANOMALY = "No Anomaly"
_BIN_DESC = ", ".join(f"{name} ({lo:g}–{hi:g}σ)" for name, lo, hi in _BINS)
_QUESTION = (
    "How large is the anomaly's peak deviation, in multiples of the channel's "
    f"standard deviation? Bins: {_BIN_DESC}. "
    "Choose 'No Anomaly' if no anomaly is present."
)


class Magnitude(QuestionTemplate):
    category = "magnitude"
    tier = 2
    min_channels = 1

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
            raise ValueError(f"Magnitude needs >=1 channel, got {n_channels}")
        scene = build_scene(n_channels, length, rng)

        # 5 options: 4 magnitude buckets + No Anomaly.
        choices = list(range(len(_BINS))) + [None]
        gold_bucket = choices[int(rng.integers(0, len(choices)))]

        injections = []
        if gold_bucket is None:
            gold_label = _NO_ANOMALY
        else:
            name, lo, hi = _BINS[gold_bucket]
            gold_label = name
            magnitude = float(rng.uniform(lo, hi))
            target = scene.names[int(rng.integers(0, n_channels))]
            width = int(rng.integers(MIN_WIDTH, min(MAX_WIDTH, length) + 1))
            t_start = int(rng.integers(0, length - width + 1))
            spec = InjectionSpec(channel=target, t_start=t_start, t_end=t_start + width,
                                 kind="level_shift", magnitude=magnitude)
            scene.channels[target] = inject(scene.channels[target], spec, rng)
            injections.append(spec)

        options = [name for name, _, _ in _BINS] + [_NO_ANOMALY]
        rng.shuffle(options)

        series = {name: arr.astype(np.float32).tolist() for name, arr in scene.channels.items()}

        return Row(
            series=series,
            question=_QUESTION,
            options=options,
            answer=gold_label,
            semantic_class=gold_label,
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        return gold
