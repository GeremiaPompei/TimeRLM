"""Localize-all-channels (Tier III, multi-channel evidence-grounded):
report every anomaly across C channels as `{channel, start, end}` events.

K (anomalies per row) is sampled uniformly in `{0, 1, 2, 3}` per plan
§3.4 — K=0 is essential so a precision-style false-positive penalty is
meaningful. Each anomaly lives on a *distinct* channel (per the v2
`_multi_injection.place_k_anomalies` contract).

Channel set in v2 is fixed at C=16 (plan §4 + locked decision). The
template is registered with `min_channels=16`; ablations on smaller
channel subsets happen post-hoc at eval time by filtering events.

Gold payload always contains an `anomalies` list (possibly empty) plus
`_L`. Events are sorted by `(channel, start)` for stable equality in
tests and reproducibility.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..difficulty import Difficulty
from ..scene import build_scene
from ._multi_injection import place_k_anomalies
from .base import QuestionTemplate, Row

_CATEGORY = "localize_all_channels"

_MAX_K: int = 3

_QUESTION_TEMPLATE = (
    "The time series has multiple channels. Find every anomaly across "
    "all channels. Respond with exactly one JSON object on a single "
    "line, no other text:\n"
    '{"anomalies": [{"channel": "<name>", "start": <int>, "end": <int>}, ...]}\n'
    "Report an empty list `\"anomalies\": []` if no anomalies are present. "
    "`channel` must match a channel name from the input verbatim. `start` is "
    "inclusive, `end` is exclusive. (Here L = __LENGTH__, C = __N_CHANNELS__.)"
)


class LocalizeAllChannels(QuestionTemplate):
    category = _CATEGORY
    tier = 3
    min_channels = 16

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
        if n_channels < self.min_channels:
            raise ValueError(
                f"LocalizeAllChannels needs >={self.min_channels} channels, got {n_channels}"
            )
        difficulty = difficulty or Difficulty()

        scene = build_scene(n_channels, length, rng, base_noise_weights=difficulty.noise_weights)
        # Benchmark default: k ~ uniform{0..max_k}, so ~25% of rows carry empty gold
        # (the no-anomaly case). RL post-training's first phase sets
        # ANOMALYXL_ALLCH_MIN_K=1 to drop those rows: from a cold start they collapse
        # the policy to always-"anomalies": [] (71% empty answers). See
        # configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml.
        min_k = int(os.getenv("ANOMALYXL_ALLCH_MIN_K", "0"))
        k = int(rng.integers(min_k, difficulty.max_k + 1))
        injections = place_k_anomalies(scene, k, rng, difficulty=difficulty)

        # Sort events by (channel, start) so gold is order-stable.
        events = sorted(
            (
                {
                    "channel": spec.channel,
                    "start": int(spec.t_start),
                    "end": int(spec.t_end),
                }
                for spec in injections
            ),
            key=lambda e: (e["channel"], e["start"]),
        )
        payload = {
            "anomalies": events,
            "_L": int(length),
        }

        series = {
            name: arr.astype(np.float32).tolist()
            for name, arr in scene.channels.items()
        }
        gold = f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}"

        question = (
            _QUESTION_TEMPLATE
            .replace("__LENGTH__", str(length))
            .replace("__N_CHANNELS__", str(n_channels))
        )

        return Row(
            series=series,
            question=question,
            options=[],
            answer=gold,
            semantic_class=f"{_CATEGORY}_K{k}_L{length}",
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        payload = _parse_gold_payload(gold)
        k = len(payload.get("anomalies", []))
        L = int(payload.get("_L", 0))
        return f"{_CATEGORY}_K{k}_L{L}"


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a localize_all_channels gold: {gold!r}")
    return json.loads(gold[len(prefix):])
