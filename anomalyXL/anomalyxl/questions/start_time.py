"""Start Time (Tier II): which quarter of the series does the anomaly start in?

Options: 5 = the 4 ordinal quarters of `[0, L)` + `"No Anomaly"`. Bucket
labels bake in the concrete index range so the option text is unambiguous
regardless of `L`. Gold is picked first; `t_start` is sampled from the gold
bucket.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..scene import build_scene
from ._time_buckets import QUARTER_LABELS_START, bucket_bounds, bucket_label
from .base import QuestionTemplate, Row

_NO_ANOMALY = "No Anomaly"
_QUESTION = (
    "In which time range does the anomaly begin? "
    "Choose 'No Anomaly' if no anomaly is present."
)


class StartTime(QuestionTemplate):
    category = "start_time"
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
            raise ValueError(f"StartTime needs >=1 channel, got {n_channels}")
        scene = build_scene(n_channels, length, rng)

        # 5 options: 4 quarter buckets + "No Anomaly".
        choices = list(range(len(QUARTER_LABELS_START))) + [None]
        gold_bucket = choices[int(rng.integers(0, len(choices)))]
        injections = []
        if gold_bucket is None:
            gold_label = _NO_ANOMALY
        else:
            gold_label = bucket_label(QUARTER_LABELS_START[gold_bucket], length, gold_bucket)
            target = scene.names[int(rng.integers(0, n_channels))]
            lo, hi = bucket_bounds(length, gold_bucket)
            spec = _sample_injection_with_start_in(length, rng, lo, hi, target)
            scene.channels[target] = inject(scene.channels[target], spec, rng)
            injections.append(spec)

        options = [bucket_label(p, length, i) for i, p in enumerate(QUARTER_LABELS_START)]
        options.append(_NO_ANOMALY)
        rng.shuffle(options)

        series = {name: arr.astype(np.float32).tolist() for name, arr in scene.channels.items()}

        return Row(
            series=series,
            question=_QUESTION,
            options=options,
            answer=gold_label,
            semantic_class=QUARTER_LABELS_START[gold_bucket] if gold_bucket is not None else _NO_ANOMALY,
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        if gold == _NO_ANOMALY:
            return _NO_ANOMALY
        for prefix in QUARTER_LABELS_START:
            if gold.startswith(prefix + " "):
                return prefix
        raise ValueError(f"Cannot map gold {gold!r} to a StartTime semantic class")


def _sample_injection_with_start_in(L: int, rng: np.random.Generator, lo: int, hi: int, target: str):
    """Pick an anomaly whose `t_start` falls in `[lo, hi)`. Retries-free: we
    sample the kind and width first, then pin `t_start` inside the bucket."""
    from ..anomalies import InjectionSpec, MIN_WIDTH, MAX_WIDTH
    kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
    max_w = min(MAX_WIDTH, L - lo)
    if max_w < MIN_WIDTH:
        # Bucket starts so close to L that no minimum-width window fits;
        # widen the start to the latest position that admits a min-width window.
        t_start = L - MIN_WIDTH
        width = MIN_WIDTH
    else:
        width = int(rng.integers(MIN_WIDTH, max_w + 1))
        latest_start = min(hi - 1, L - width)
        latest_start = max(lo, latest_start)
        t_start = int(rng.integers(lo, latest_start + 1))
    t_end = t_start + width
    magnitude = float(rng.uniform(2.0, 6.0))
    return InjectionSpec(channel=target, t_start=t_start, t_end=t_end, kind=kind, magnitude=magnitude)
