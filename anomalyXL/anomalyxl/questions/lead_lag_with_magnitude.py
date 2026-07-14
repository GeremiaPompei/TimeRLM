"""Lead-lag-with-magnitude (Tier III paired, evidence-grounded):
identify the temporal relationship between anomalies in two
single-channel sub-scenes A and B, AND quantify the lag.

Gold envelope:
``lead_lag_with_magnitude:{"direction": <"lead"|"lag"|"independent">,
                          "lag_samples": <int>, "_L": <int>}``.

`lag_samples` is always non-negative — it's the *magnitude* of the
offset between A's and B's anomaly start indices. The direction tag
disambiguates sign:
- ``"lead"``: A's anomaly precedes B's by ``lag_samples`` (i.e. A is
  the leading indicator).
- ``"lag"``: A's anomaly follows B's by ``lag_samples`` (A is the
  lagging indicator).
- ``"independent"``: both sides contain anomalies in disjoint windows;
  ``lag_samples`` is ignored.

Direction is sampled 1/3 / 1/3 / 1/3 per plan §3.5; for lead/lag the
magnitude is uniform in ``[1, L/4]``. C is fixed at 2 (one channel per
side) — keeps the prompt narrow and the gold unambiguous.

Reuses ``_paired.build_paired_scenes`` for the A/B scaffold and
``_paired.inject_for_coupling``'s ``independent`` branch for the
no-relationship case; the coupled cases inline the per-side injection
so we can pin ``lag_samples`` exactly without going through
``make_coupling``'s own lag sampler.
"""

from __future__ import annotations

import json

import numpy as np

from ..anomalies import (
    ANOMALY_KINDS,
    CouplingSpec,
    InjectionSpec,
    MAX_WIDTH,
    MIN_WIDTH,
    inject,
)
from ..difficulty import Difficulty
from .base import QuestionTemplate, Row
from ._paired import build_paired_scenes, inject_for_coupling

_CATEGORY = "lead_lag_with_magnitude"

_DIRECTIONS: tuple[str, ...] = ("lead", "lag", "independent")

_QUESTION_TEMPLATE = (
    "Two single-channel sub-series are shown, with channels prefixed "
    "`A_` and `B_`. Both contain an anomaly (or independent anomalies). "
    "Is A a leading or lagging indicator of B, and by how many samples? "
    "Respond with exactly one JSON object on a single line, no other text:\n"
    '{"direction": "lead"|"lag"|"independent", "lag_samples": <int>}\n'
    "`lag_samples` is the non-negative magnitude (in sample indices) of "
    "the offset between A's and B's anomaly start; it is ignored in the "
    "independent case. "
    "Two anomalies are considered independent when they occupy unrelated "
    "windows with no causal alignment; by construction, coupled lead/lag "
    "offsets never exceed L/4 samples. (Here L = __LENGTH__.)"
)


class LeadLagWithMagnitude(QuestionTemplate):
    category = _CATEGORY
    tier = 3
    min_channels = 2

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
        difficulty: "Difficulty | None" = None,
    ) -> Row:
        if n_channels < self.min_channels:
            raise ValueError(
                f"LeadLagWithMagnitude needs >={self.min_channels} channels, got {n_channels}"
            )
        difficulty = difficulty or Difficulty()

        scene_a, scene_b = build_paired_scenes(n_channels, length, rng, paired_split, difficulty=difficulty)
        direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]

        if direction == "independent":
            coupling = CouplingSpec(kind="independent", lag=0)
            injections, coupling = inject_for_coupling(scene_a, scene_b, coupling, rng, difficulty=difficulty)
            lag_samples = 0
        else:
            # Pin lag magnitude in [1, L/4]; sign comes from direction.
            max_lag = max(1, int(length * difficulty.lag_max_fraction))
            # Guard against the (theoretical) case where MAX_WIDTH eats
            # most of the series — ensure we leave room for both windows.
            max_lag = min(max_lag, length - MAX_WIDTH)
            if max_lag < 1:
                raise ValueError(
                    f"L={length} is too short for any lag with MAX_WIDTH={MAX_WIDTH}"
                )
            lag_samples = int(rng.integers(1, max_lag + 1))
            signed_lag = lag_samples if direction == "lead" else -lag_samples
            coupling = CouplingSpec(
                kind="coupled_lag_pos" if direction == "lead" else "coupled_lag_neg",
                lag=signed_lag,
            )
            injections, coupling = inject_for_coupling(scene_a, scene_b, coupling, rng, difficulty=difficulty)
            # Recover the actual lag in case `inject_for_coupling` re-clamped
            # the coupling (e.g. degraded to coupled_simul on a too-short series).
            if coupling.kind == "coupled_simul":
                # Re-clamp surfaced as a direction collapse — this shouldn't
                # happen at the L values we generate, but be precise about it.
                raise RuntimeError(
                    f"inject_for_coupling degraded to simultaneous at L={length}, "
                    f"lag={signed_lag}. Tighten max_lag bound."
                )
            lag_samples = abs(coupling.lag)

        payload = {
            "direction": direction,
            "lag_samples": int(lag_samples),
            "_L": int(length),
        }

        series = {
            **{name: arr.astype(np.float32).tolist() for name, arr in scene_a.channels.items()},
            **{name: arr.astype(np.float32).tolist() for name, arr in scene_b.channels.items()},
        }
        gold = f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}"

        return Row(
            series=series,
            question=_QUESTION_TEMPLATE.replace("__LENGTH__", str(length)),
            options=[],
            answer=gold,
            semantic_class=f"{_CATEGORY}_{direction}_L{length}",
            injections=injections,
            coupling=coupling,
        )

    def semantic_class(self, gold: str) -> str:
        payload = _parse_gold_payload(gold)
        direction = payload["direction"]
        L = int(payload.get("_L", 0))
        return f"{_CATEGORY}_{direction}_L{L}"


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a lead_lag_with_magnitude gold: {gold!r}")
    return json.loads(gold[len(prefix):])
