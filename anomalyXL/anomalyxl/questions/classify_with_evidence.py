"""Classify-with-evidence (Tier II, evidence-grounded): identify the
anomaly's kind AND its window in a single JSON object.

Gold format: ``classify_with_evidence:{"kind": <label>, "start": int,
"end": int, "_L": int}``. The composite scorer
``kind_x_iou = kind_correct × iou`` is what enforces the grounding
requirement — a right label with the wrong window scores ``iou``, a
wrong label scores 0 regardless.

`kind` labels reuse the ARFBench wording from
`categorization.py:_KIND_TO_LABEL` so the kind vocabulary is identical
across categories.
"""

from __future__ import annotations

import json

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..difficulty import Difficulty
from ..scene import build_scene
from .base import QuestionTemplate, Row
from .categorization import _KIND_TO_LABEL

_CATEGORY = "classify_with_evidence"

# Comma-joined ARFBench labels, deterministic order — appears in the
# question text so the model knows the legal `kind` vocabulary.
_KIND_LABELS_LIST = ", ".join(f'"{label}"' for label in _KIND_TO_LABEL.values())

_QUESTION_TEMPLATE = (
    "Identify the anomaly in the time series and its window. Respond "
    "with exactly one JSON object on a single line, no other text:\n"
    '{"kind": <label>, "start": <int>, "end": <int>}\n'
    f"`kind` must be one of: {_KIND_LABELS_LIST}. `start` is inclusive, "
    "`end` is exclusive. "
    "Each kind is defined as follows: "
    "a `Level Shift` is a sustained step in mean — values jump and remain shifted across the window; "
    "a `Transient Spike` is a brief impulse — one or a few samples diverge sharply, then return to baseline; "
    "a `Change in Variance` is a sustained increase (or decrease) in noise amplitude over a window with the mean unchanged; "
    "a `Change in Seasonality` is a sustained change in the periodic component's amplitude, frequency, or phase across a window; "
    "a `Change in Trend` is a slope change starting at the window — values drift up or down and remain on the new path afterwards. "
    "(Here L = __LENGTH__.)"
)


class ClassifyWithEvidence(QuestionTemplate):
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
            raise ValueError(f"ClassifyWithEvidence needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

        scene = build_scene(n_channels, length, rng, base_noise_weights=difficulty.noise_weights)
        target = scene.names[int(rng.integers(0, n_channels))]
        kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        spec = make_injection(
            length, rng, kind=kind, channel=target,
            min_width=difficulty.width_range[0], max_width=difficulty.width_range[1],
            magnitude_range=difficulty.magnitude_range,
        )
        scene.channels[target] = inject(scene.channels[target], spec, rng)

        kind_label = _KIND_TO_LABEL[kind]
        payload = {
            "kind": kind_label,
            "start": int(spec.t_start),
            "end": int(spec.t_end),
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
            semantic_class=f"{_CATEGORY}_{kind}_L{length}",
            injections=[spec],
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        payload = _parse_gold_payload(gold)
        label = payload["kind"]
        # Map ARFBench label back to the snake_case kind for the bin key.
        kind_key = _LABEL_TO_KIND.get(label, label)
        L = int(payload.get("_L", 0))
        return f"{_CATEGORY}_{kind_key}_L{L}"


_LABEL_TO_KIND: dict[str, str] = {v: k for k, v in _KIND_TO_LABEL.items()}


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a classify_with_evidence gold: {gold!r}")
    return json.loads(gold[len(prefix):])
