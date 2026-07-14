"""Measure-magnitude (Tier II, evidence-grounded): report the anomaly's
peak deviation in σ-units of the baseline (pre-injection) channel std.

Implementation contract:
- Always inject a single ``level_shift`` so peak deviation = exactly
  ``magnitude × pre_std`` analytically (no envelope, no decay, no
  stochastic peak).
- ``magnitude`` is sampled **log-uniformly** in ``[1.0, 100.0]`` —
  matches the ``acc_at_rel_*`` band spacing and stresses both
  near-floor (~1σ) and far (~100σ) ranges.
- The gold ``magnitude_sigma`` is computed from the *actual* arrays:
  ``peak / pre_std`` where ``pre_std`` is the empirical std of the
  channel before injection. With ``scene._normalize_unit`` z-norming
  the channel, ``pre_std`` is ≈1, but we still compute it explicitly
  so the gold tracks float-precision drift instead of rounding to a
  magic scalar.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..anomalies import InjectionSpec, MIN_WIDTH, MAX_WIDTH, inject
from ..difficulty import Difficulty
from ..scene import build_scene
from .base import QuestionTemplate, Row

_CATEGORY = "measure_magnitude"

_MAGNITUDE_LO: float = 1.0
_MAGNITUDE_HI: float = 100.0

_QUESTION_TEMPLATE = (
    "An anomaly is present in the time series. Report its peak deviation "
    "in units of the channel's baseline (pre-anomaly) standard deviation "
    "(σ). Respond with exactly one JSON object on a single line, no other "
    "text:\n"
    '{"magnitude_sigma": <float>}\n'
    "Use a positive value (the deviation magnitude, not its sign). "
    "(Here L = __LENGTH__.)"
)


class MeasureMagnitude(QuestionTemplate):
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
            raise ValueError(f"MeasureMagnitude needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

        scene = build_scene(n_channels, length, rng, base_noise_weights=difficulty.noise_weights)
        target = scene.names[int(rng.integers(0, n_channels))]

        magnitude = _sample_log_uniform(rng, difficulty.magnitude_log_range[0], difficulty.magnitude_log_range[1])
        max_w = min(difficulty.width_range[1], length)
        width = int(rng.integers(difficulty.width_range[0], max(difficulty.width_range[0] + 1, max_w + 1)))
        latest_start = max(0, length - width)
        t_start = int(rng.integers(0, latest_start + 1))
        t_end = t_start + width
        spec = InjectionSpec(
            channel=target,
            t_start=t_start,
            t_end=t_end,
            kind="level_shift",
            magnitude=float(magnitude),
        )

        pre_arr = scene.channels[target]
        pre_std = float(pre_arr.std())
        if pre_std < 1e-9:
            # Defensive: build_scene z-norms so std≈1, but guard the
            # division for the (unreachable on real data) degenerate path.
            pre_std = 1.0

        scene.channels[target] = inject(pre_arr, spec, rng)

        # Empirical peak from the actual injected arrays — robust to any
        # future change in `_level_shift`.
        delta = scene.channels[target] - pre_arr
        peak = float(np.max(np.abs(delta)))
        magnitude_sigma = peak / pre_std

        payload = {
            "magnitude_sigma": round(magnitude_sigma, 6),
            "_L": int(length),
        }

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
            semantic_class=f"{_CATEGORY}_L{length}",
            injections=[spec],
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        payload = _parse_gold_payload(gold)
        L = int(payload.get("_L", 0))
        return f"{_CATEGORY}_L{L}"


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw `x ~ exp(U(log lo, log hi))`. lo and hi must be positive."""
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return math.exp(float(rng.uniform(log_lo, log_hi)))


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a measure_magnitude gold: {gold!r}")
    return json.loads(gold[len(prefix):])
