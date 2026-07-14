"""Generator-backed v1 taskset (Mode A).

Replaces the static parquet with a lazy, on-the-fly sample stream from the
`anomalyxl` generator. The taskset's `load_tasks` returns `_GeneratedTasks`, a
lazy `Sequence` over a fixed index space (`num_tasks`): each index maps
deterministically (via blake2b on `run_seed|i|salt`) to a generator cell and a
sample seed, so `tasks[i]` is reproducible and `run_seed` rotates the whole
stream for fresh data per run.

A single fixed difficulty θ is pinned across the grid; the per-task knobs ride on
the rollout record (`GeneratedAnomalyXLTask`). Generation is cheap and
deterministic, so `setup` regenerates the series from the task's fields rather
than caching it.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
from pydantic_config import BaseConfig
from anomalyxl.difficulty import Difficulty
from anomalyxl.pipeline import SampleSpec, _build_row


class GeneratorCell(BaseConfig):
    category: str
    channels: int
    lengths: list[int]


class GeneratorConfig(BaseConfig):
    num_tasks: int = 100_000
    """Size of the lazy index space the taskset draws from."""
    run_seed: int = 0
    """Rotates the whole deterministic stream — bump for fresh data per run."""
    cells: list[GeneratorCell] = [
        GeneratorCell(category="localize", channels=1, lengths=[16384, 32768, 65536, 131072]),
        GeneratorCell(category="classify_with_evidence", channels=1, lengths=[16384, 32768, 65536, 131072]),
        GeneratorCell(category="measure_magnitude", channels=1, lengths=[16384, 32768, 65536, 131072]),
        GeneratorCell(category="localize_all_channels", channels=16, lengths=[32768, 131072]),
        GeneratorCell(category="lead_lag_with_magnitude", channels=2, lengths=[32768, 131072]),
    ]
    # Fixed difficulty θ.
    magnitude_range: tuple[float, float] = (2.0, 6.0)
    width_range: tuple[int, int] = (8, 256)
    magnitude_log_range: tuple[float, float] = (1.0, 100.0)
    noise_weights: dict | None = None
    max_k: int = 3
    lag_max_fraction: float = 0.25
    width_frac_range: tuple[float, float] | None = None
    """RELATIVE anomaly width: when set, each task's absolute `width_range` is derived
    as this fraction of its series length (floored at 8 samples, capped at length/8).
    Decouples relative width from length — with the absolute default, relative width
    silently shrinks 8x from 16k to 131k series. Keep the upper fraction small
    (≲ 0.05): a whole-series guess earns IoU = width/length > 0, so large fractions
    raise the guessing baseline and erode the reward's exact zero on failure.
    None = absolute `width_range` (the AnomalyXL default)."""


def _theta(cfg: GeneratorConfig) -> Difficulty:
    return Difficulty(
        magnitude_range=cfg.magnitude_range,
        width_range=cfg.width_range,
        magnitude_log_range=cfg.magnitude_log_range,
        noise_weights=cfg.noise_weights,
        max_k=cfg.max_k,
        lag_max_fraction=cfg.lag_max_fraction,
    )


def _seed(run_seed: int, i: int, salt: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{run_seed}|{i}|{salt}".encode(), digest_size=8).digest(), "big"
    )


class _GeneratedTasks:
    """Lazy `Sequence` over the generator's index space — `tasks[i]` synthesizes
    sample `i` on demand, deterministically."""

    def __init__(self, cfg: GeneratorConfig, note: str) -> None:
        self.cfg = cfg
        self.note = note
        self.combos = [(c.category, c.channels, L) for c in cfg.cells for L in c.lengths]
        self.base_theta = _theta(cfg)

    def __len__(self) -> int:
        return self.cfg.num_tasks

    def _resolve_theta(self, category: str, length: int) -> Difficulty:
        """Per-task θ: the fixed base difficulty, with relative width resolved against
        this task's length. `width_frac_range` wins over any absolute `width_range`;
        the resolved ABSOLUTE range is what rides on the task record, so downstream
        consumers see the truth."""
        theta = self.base_theta
        frac = self.cfg.width_frac_range
        if frac is not None:
            lo = max(8, min(int(frac[0] * length), length // 8))
            hi = max(lo + 1, min(int(frac[1] * length), length // 8))
            theta = dataclasses.replace(theta, width_range=(lo, hi))
        return theta

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        from .taskset import GeneratedAnomalyXLTask

        rng = np.random.default_rng(_seed(self.cfg.run_seed, i, "pick"))
        category, channels, length = self.combos[int(rng.integers(0, len(self.combos)))]
        paired_split = (0.5, 0.5) if category == "lead_lag_with_magnitude" else None
        sample_seed = _seed(self.cfg.run_seed, i, "sample")
        theta = self._resolve_theta(category, length)
        spec = SampleSpec(
            seed=sample_seed,
            length=length,
            n_channels=channels,
            category=category,
            difficulty=theta,
            paired_split=paired_split,
        )
        row = _build_row(spec)
        return GeneratedAnomalyXLTask(
            idx=i,
            prompt=f"{self.note}\n\n{row.question}",
            answer=row.answer,
            seed=sample_seed,
            length=length,
            n_channels=channels,
            category=category,
            paired_split=paired_split,
            difficulty=dataclasses.asdict(theta),
        )
