"""Localize (Tier II, evidence-grounded): does the series contain an
anomaly, and if so, what is its window?

The model commits to *both* presence and grounding in a single JSON
object: `{"present": bool, "start": int, "end": int}`. Gold is sampled
80% present / 20% absent — the absent case is essential so a
precision-style false-positive penalty exists, but a uniform 50/50
split over-weights the trivial "say no" baseline and dilutes the
localization signal. 20% absent keeps the precision pressure while
leaving 80% of rows scoring grounded IoU.

Gold envelope is the AnomalyXL convention:
``f"localize:{json.dumps({...})}"`` where the payload always carries
`_L` so the env scorer can compute MAE fractions without re-reading
the parquet `length` column.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..difficulty import Difficulty
from ..scene import build_scene
from .base import QuestionTemplate, Row

_CATEGORY = "localize"

# Probability the row gets an injected anomaly (present=True). The remainder
# ((1 - _P_PRESENT)) carries the hard-negative present=False case. The benchmark
# default is 0.8 / 0.2: it keeps the false-positive penalty meaningful while
# making the IoU/MAE metrics the dominant signal.
#
# RL post-training overrides this for its first phase (ANOMALYXL_P_PRESENT=1.0):
# from a cold start the 20% absent rows are a low-effort attractor and the policy
# collapses to always-"present": false (56% absent answers vs 25% gold). Training
# all-present kills that exit, and negatives are staged back in mid-run to
# re-teach calibration. See configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml.
# Read at import time, so it must be exported in the process's launch environment.
_P_PRESENT: float = float(os.getenv("ANOMALYXL_P_PRESENT", "0.8"))

_QUESTION_TEMPLATE = (
    "Does the time series contain an anomaly, and if so where? Respond "
    "with exactly one JSON object on a single line, no other text:\n"
    '{"present": true|false, "start": <int>, "end": <int>}\n'
    "If \"present\" is false, \"start\" and \"end\" are ignored. If true, "
    "report the inclusive start and exclusive end integer sample indices "
    "of the anomaly. (Here L = __LENGTH__.)"
)


class Localize(QuestionTemplate):
    category = _CATEGORY
    tier = 2
    min_channels = 1

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
        difficulty: "Difficulty | None" = None,
    ) -> Row:
        del paired_split
        if n_channels < 1:
            raise ValueError(f"Localize needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

        scene = build_scene(n_channels, length, rng, base_noise_weights=difficulty.noise_weights)
        present = bool(rng.uniform() < _P_PRESENT)

        injections = []
        if present:
            target = scene.names[int(rng.integers(0, n_channels))]
            kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
            spec = make_injection(
                length, rng, kind=kind, channel=target,
                min_width=difficulty.width_range[0], max_width=difficulty.width_range[1],
                magnitude_range=difficulty.magnitude_range,
            )
            scene.channels[target] = inject(scene.channels[target], spec, rng)
            injections.append(spec)
            payload = {
                "present": True,
                "start": int(spec.t_start),
                "end": int(spec.t_end),
                "_L": int(length),
            }
            semantic = f"localize_present_L{length}"
        else:
            payload = {"present": False, "_L": int(length)}
            semantic = f"localize_absent_L{length}"

        series = {
            name: arr.astype(np.float32).tolist()
            for name, arr in scene.channels.items()
        }
        gold = f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}"

        return Row(
            series=series,
            question=_QUESTION_TEMPLATE.replace("__LENGTH__", str(length)),
            options=[],
            answer=gold,
            semantic_class=semantic,
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        payload = _parse_gold_payload(gold)
        L = int(payload.get("_L", 0))
        return f"localize_present_L{L}" if payload.get("present") else f"localize_absent_L{L}"


def _parse_gold_payload(gold: str) -> dict:
    """Recover the JSON payload from a ``localize:<json>`` gold string."""
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a localize gold: {gold!r}")
    return json.loads(gold[len(prefix):])
