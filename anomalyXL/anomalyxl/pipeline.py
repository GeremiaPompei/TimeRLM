"""Orchestration: `SampleSpec` -> `Row` for one sample, plus grid construction.

`generate` is a generator over `Row`s; `cli.py` consumes it and writes
parquet incrementally so memory stays bounded for the full sweep.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .difficulty import Difficulty
from .questions import TEMPLATES, Row

__all__ = ["SampleSpec", "Row", "generate", "specs_from_grid", "row_id"]


@dataclass(frozen=True)
class SampleSpec:
    """One sample's coordinates. The `seed` is *absolute* and fully determines
    the synthesized scene + question + gold."""

    seed: int
    length: int
    n_channels: int
    category: str
    paired_split: tuple[float, float] | None = None
    difficulty: Difficulty | None = None


def _rng_for(spec: SampleSpec) -> np.random.Generator:
    return np.random.default_rng(spec.seed)


def row_id(spec: SampleSpec) -> str:
    """Deterministic short ID for a (seed, L, C, category) cell."""
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{spec.seed}|{spec.length}|{spec.n_channels}|{spec.category}".encode())
    return h.hexdigest()


def _build_row(spec: SampleSpec) -> Row:
    if spec.category not in TEMPLATES:
        raise KeyError(
            f"Unknown category {spec.category!r}; available: {sorted(TEMPLATES)}"
        )
    template = TEMPLATES[spec.category]()
    rng = _rng_for(spec)
    kwargs = dict(length=spec.length, n_channels=spec.n_channels, rng=rng, paired_split=spec.paired_split)
    if spec.difficulty is not None and "difficulty" in inspect.signature(template.build).parameters:
        kwargs["difficulty"] = spec.difficulty
    return template.build(**kwargs)


def generate(specs: Sequence[SampleSpec]) -> Iterator[tuple[SampleSpec, Row]]:
    """Yield `(spec, row)` pairs in the input order. No batching, no parallelism
    in v1 — `cli.py` writes to parquet incrementally so memory stays bounded."""
    for spec in specs:
        yield spec, _build_row(spec)


def specs_from_grid(
    lengths: Sequence[int],
    channels: Sequence[int],
    per_cell: int,
    categories: Sequence[str],
    seed: int,
) -> list[SampleSpec]:
    """Build the full `L × C × category × per_cell` spec list.

    Each sample's seed is derived from `(global_seed, length, n_channels,
    category, index_in_cell)` so adding categories or per-cell rows later
    doesn't reshuffle the existing ones.

    Cells where `n_channels < template.min_channels` are filtered out
    silently (e.g. Identification needs `C >= 3`).
    """
    out: list[SampleSpec] = []
    base = np.random.default_rng(seed)
    # One stable per-cell seed offset, derived from the global seed only.
    cell_seed_offset = int(base.integers(1, 2**31 - 1))
    for length in lengths:
        for n_channels in channels:
            for category in categories:
                if category not in TEMPLATES:
                    raise KeyError(f"Unknown category {category!r}; available: {sorted(TEMPLATES)}")
                if n_channels < TEMPLATES[category].min_channels:
                    continue
                tpl = TEMPLATES[category]
                paired_split = (0.5, 0.5) if tpl.tier == 3 else None
                for i in range(per_cell):
                    h = hashlib.blake2b(digest_size=4)
                    h.update(f"{seed}|{cell_seed_offset}|{length}|{n_channels}|{category}|{i}".encode())
                    sample_seed = int.from_bytes(h.digest(), "big")
                    out.append(SampleSpec(
                        seed=sample_seed,
                        length=length,
                        n_channels=n_channels,
                        category=category,
                        paired_split=paired_split,
                    ))
    return out
