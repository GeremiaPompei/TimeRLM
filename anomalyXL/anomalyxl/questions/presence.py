"""Presence (Tier I): yes/no — is any anomaly present in any channel?

Gold is uniform over {"Yes", "No"}. "Yes" injects one anomaly in one random
channel; "No" skips injection entirely. Options are always `["Yes", "No"]`
(deterministically shuffled).
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..scene import build_scene
from .base import QuestionTemplate, Row

_QUESTION = "Is any anomaly present in the time series?"


class Presence(QuestionTemplate):
    category = "presence"
    tier = 1
    min_channels = 1

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
    ) -> Row:
        del paired_split  # tier I: single scene only
        if n_channels < 1:
            raise ValueError(f"Presence needs >=1 channel, got {n_channels}")

        scene = build_scene(n_channels, length, rng)
        gold = "Yes" if rng.uniform() < 0.5 else "No"

        injections = []
        if gold == "Yes":
            target = scene.names[int(rng.integers(0, n_channels))]
            kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
            spec = make_injection(length, rng, kind=kind, channel=target)
            scene.channels[target] = inject(scene.channels[target], spec, rng)
            injections.append(spec)

        options = ["Yes", "No"]
        rng.shuffle(options)

        series = {name: arr.astype(np.float32).tolist() for name, arr in scene.channels.items()}

        return Row(
            series=series,
            question=_QUESTION,
            options=options,
            answer=gold,
            semantic_class=gold,
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        return gold
