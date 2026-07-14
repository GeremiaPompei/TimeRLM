"""Identification (Tier II): which subset of three named channels is anomalous?

Three channels are picked from the full scene and named in the question.
Gold subset size ∈ {0, 1, 2, 3} → distinct semantic bins. Options are always
5: two singletons (one is gold if gold is a singleton), one pair (gold if
gold is a pair), the only triplet `{A, B, C}`, and `"None"`.

Channels *outside* the named three may also be anomalous (distractors). This
is why difficulty scales with `C`: at C=4 there's at most 1 distractor; at
C=64 there can be many.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection, inject
from ..scene import build_scene
from .base import QuestionTemplate, Row

_NONE = "None"

# Per-non-picked-channel probability of being a distractor anomaly. Calibrated
# so C=4 is near-empty (~0.2 distractors) and C=64 is busy (~12 distractors).
_DISTRACTOR_P: float = 0.2


def _fmt(channels: list[str]) -> str:
    """Sorted, comma-joined channel label used as option text."""
    return ", ".join(sorted(channels)) if channels else _NONE


class Identification(QuestionTemplate):
    category = "identification"
    tier = 2
    min_channels = 3

    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
    ) -> Row:
        del paired_split
        if n_channels < 3:
            raise ValueError(
                f"Identification needs >=3 channels, got {n_channels}; "
                "filter incompatible cells in specs_from_grid."
            )

        scene = build_scene(n_channels, length, rng)
        all_names = scene.names

        picked_idx = rng.choice(n_channels, size=3, replace=False)
        picked = sorted(all_names[i] for i in picked_idx)
        non_picked = [n for n in all_names if n not in set(picked)]

        # Gold subset size uniform over {0,1,2,3}.
        size = int(rng.integers(0, 4))
        if size == 0:
            gold_subset: list[str] = []
        else:
            gold_idx = rng.choice(3, size=size, replace=False)
            gold_subset = sorted(picked[i] for i in gold_idx)
        gold_label = _fmt(gold_subset)

        # Inject anomalies in gold subset.
        injections = []
        for ch in gold_subset:
            kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
            spec = make_injection(length, rng, kind=kind, channel=ch)
            scene.channels[ch] = inject(scene.channels[ch], spec, rng)
            injections.append(spec)

        # Distractors outside the picked subset; per-channel Bernoulli.
        for ch in non_picked:
            if rng.uniform() < _DISTRACTOR_P:
                kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
                spec = make_injection(length, rng, kind=kind, channel=ch)
                scene.channels[ch] = inject(scene.channels[ch], spec, rng)
                injections.append(spec)

        options = _build_options(picked, gold_subset, rng)

        question = (
            f"Among channels {picked[0]}, {picked[1]}, and {picked[2]}, "
            "which subset (if any) is anomalous? Other channels in the series may "
            "also be anomalous, but only these three are scored."
        )

        series = {name: arr.astype(np.float32).tolist() for name, arr in scene.channels.items()}

        return Row(
            series=series,
            question=question,
            options=options,
            answer=gold_label,
            semantic_class=str(len(gold_subset)),
            injections=injections,
            coupling=None,
        )

    def semantic_class(self, gold: str) -> str:
        if gold == _NONE:
            return "0"
        return str(len([s for s in gold.split(", ") if s]))


def _build_options(picked: list[str], gold_subset: list[str], rng: np.random.Generator) -> list[str]:
    """Always 5 options: 2 singletons + 1 pair + the triplet + 'None'.

    Whatever shape the gold is, the corresponding slot in the option set is
    the gold; the others are sampled distractors. Deterministically shuffled.
    """
    a, b, c = picked  # sorted
    singletons_all = [a, b, c]
    pairs_all = [_fmt([a, b]), _fmt([a, c]), _fmt([b, c])]
    triplet = _fmt([a, b, c])
    gold_label = _fmt(gold_subset)
    size = len(gold_subset)

    if size == 1:
        # Gold is a singleton; pick one other singleton at random.
        others = [s for s in singletons_all if s != gold_label]
        rng.shuffle(others)
        singletons = sorted([gold_label, others[0]])
        pair = pairs_all[int(rng.integers(0, len(pairs_all)))]
    elif size == 2:
        # Gold is a pair; two random distinct singletons + the gold pair.
        picks = list(rng.choice(singletons_all, size=2, replace=False))
        singletons = sorted([str(p) for p in picks])
        pair = gold_label
    else:
        # size in {0, 3}: pick 2 random distinct singletons + 1 random pair.
        picks = list(rng.choice(singletons_all, size=2, replace=False))
        singletons = sorted([str(p) for p in picks])
        pair = pairs_all[int(rng.integers(0, len(pairs_all)))]

    options = singletons + [pair, triplet, _NONE]
    rng.shuffle(options)
    return options
