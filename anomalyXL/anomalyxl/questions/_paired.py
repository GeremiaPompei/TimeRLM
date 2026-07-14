"""Shared helpers for Tier III paired templates (`Correlation`, `Indicator`).

Both templates need the same plumbing: split `n_channels` into `(C_A, C_B)`,
build two scenes, sample a `CouplingSpec`, inject anomalies as the coupling
dictates, and merge the channel dicts. Only the question text + option set +
gold-derivation differ.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..anomalies import (
    ANOMALY_KINDS,
    MIN_WIDTH,
    MAX_WIDTH,
    CouplingSpec,
    InjectionSpec,
    inject,
    make_coupling,
)
from ..difficulty import Difficulty
from ..scene import Scene, build_scene

__all__ = ["split_channels", "build_paired_scenes", "inject_for_coupling"]


def split_channels(total: int, paired_split: tuple[float, float] | None) -> tuple[int, int]:
    """`(C_A, C_B)` summing to `total`. Default split is balanced; the larger
    side gets the leftover when `total` is odd."""
    if total < 2:
        raise ValueError(f"Paired templates need n_channels >= 2, got {total}")
    if paired_split is None:
        a = total // 2
        return a, total - a
    sa, sb = paired_split
    if sa <= 0 or sb <= 0:
        raise ValueError(f"paired_split components must be positive, got {paired_split}")
    a = max(1, int(round(total * sa / (sa + sb))))
    a = min(a, total - 1)
    return a, total - a


def build_paired_scenes(
    total_channels: int,
    length: int,
    rng: np.random.Generator,
    paired_split: tuple[float, float] | None,
    *,
    difficulty: "Difficulty | None" = None,
) -> tuple[Scene, Scene]:
    difficulty = difficulty or Difficulty()
    a, b = split_channels(total_channels, paired_split)
    scene_a = build_scene(a, length, rng, name_prefix="A_", base_noise_weights=difficulty.noise_weights)
    scene_b = build_scene(b, length, rng, name_prefix="B_", base_noise_weights=difficulty.noise_weights)
    return scene_a, scene_b


def _sample_window(
    L: int,
    rng: np.random.Generator,
    *,
    must_fit_after: int = 0,
    width_range: tuple[int, int] = (MIN_WIDTH, MAX_WIDTH),
) -> tuple[int, int, int]:
    """`(t_start, t_end, width)` with `t_start >= must_fit_after` and window in `[0, L)`."""
    max_w = min(width_range[1], L - must_fit_after)
    if max_w < width_range[0]:
        raise ValueError(f"Cannot fit min-width window after position {must_fit_after} in L={L}")
    width = int(rng.integers(width_range[0], max_w + 1))
    latest_start = L - width
    if latest_start < must_fit_after:
        latest_start = must_fit_after
    t_start = int(rng.integers(must_fit_after, latest_start + 1))
    return t_start, t_start + width, width


def _pick_target(scene: Scene, rng: np.random.Generator) -> str:
    return scene.names[int(rng.integers(0, len(scene.names)))]


def inject_for_coupling(
    scene_a: Scene,
    scene_b: Scene,
    coupling: CouplingSpec,
    rng: np.random.Generator,
    *,
    difficulty: "Difficulty | None" = None,
) -> tuple[list[InjectionSpec], CouplingSpec]:
    """Apply anomalies in-place on `scene_a.channels` / `scene_b.channels` per
    the coupling. Returns the list of placed `InjectionSpec`s and the
    (possibly re-clamped) `CouplingSpec`. v1 scope: one anomaly per side max.
    """
    difficulty = difficulty or Difficulty()
    kind = coupling.kind
    L = scene_a.length
    injections: list[InjectionSpec] = []

    if kind == "no_anomaly":
        return [], coupling

    if kind in {"one_sided_A", "one_sided_B"}:
        scene = scene_a if kind == "one_sided_A" else scene_b
        target = _pick_target(scene, rng)
        anomaly_kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        t_start, t_end, _ = _sample_window(L, rng, width_range=difficulty.width_range)
        spec = InjectionSpec(
            channel=target, t_start=t_start, t_end=t_end,
            kind=anomaly_kind, magnitude=float(rng.uniform(*difficulty.magnitude_range)),
        )
        scene.channels[target] = inject(scene.channels[target], spec, rng)
        injections.append(spec)
        return injections, coupling

    if kind == "independent":
        # Two anomalies, possibly different kinds. "Independent" must be
        # distinguishable from the coupled classes by the documented rule
        # (coupled offsets never exceed lag_max_fraction * L), so enforce a
        # start-to-start separation strictly greater than that bound — and
        # give either series the earlier window equally often, so temporal
        # order carries no signal about the class.
        anomaly_kind_a = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        anomaly_kind_b = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        min_gap = int(L * difficulty.lag_max_fraction) + 1
        w_early = int(rng.integers(difficulty.width_range[0], min(difficulty.width_range[1], L) + 1))
        w_late = int(rng.integers(difficulty.width_range[0], min(difficulty.width_range[1], L) + 1))
        max_early_start = L - w_late - min_gap
        if max_early_start < 0:
            raise ValueError(
                f"L={L} too short for independent pair with min_gap={min_gap}")
        t_early = int(rng.integers(0, max_early_start + 1))
        t_late = int(rng.integers(t_early + min_gap, L - w_late + 1))
        if rng.integers(0, 2) == 0:
            t_start_a, t_end_a = t_early, t_early + w_early
            t_start_b, t_end_b = t_late, t_late + w_late
        else:
            t_start_a, t_end_a = t_late, t_late + w_late
            t_start_b, t_end_b = t_early, t_early + w_early
        target_a = _pick_target(scene_a, rng)
        target_b = _pick_target(scene_b, rng)
        spec_a = InjectionSpec(channel=target_a, t_start=t_start_a, t_end=t_end_a,
                               kind=anomaly_kind_a, magnitude=float(rng.uniform(*difficulty.magnitude_range)))
        spec_b = InjectionSpec(channel=target_b, t_start=t_start_b, t_end=t_end_b,
                               kind=anomaly_kind_b, magnitude=float(rng.uniform(*difficulty.magnitude_range)))
        scene_a.channels[target_a] = inject(scene_a.channels[target_a], spec_a, rng)
        scene_b.channels[target_b] = inject(scene_b.channels[target_b], spec_b, rng)
        injections.extend([spec_a, spec_b])
        return injections, coupling

    # Coupled cases: same kind, same width; B offset by `coupling.lag`.
    anomaly_kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
    lag = coupling.lag if kind != "coupled_simul" else 0

    # Pick A's window such that B's start = A's start + lag lands in `[0, L - width]`.
    # i.e. A's t_start in `[max(0, -lag), min(L - width, L - width - lag)]`.
    max_w = min(difficulty.width_range[1], L - abs(lag))
    if max_w < difficulty.width_range[0]:
        # Too short to express the lag; degrade to simultaneous.
        lag = 0
        coupling = CouplingSpec(kind="coupled_simul", lag=0)
        max_w = min(difficulty.width_range[1], L)
    width = int(rng.integers(difficulty.width_range[0], max_w + 1))
    a_low = max(0, -lag)
    a_high = min(L - width, L - width - lag)
    if a_high < a_low:
        # Pathological; clamp to a legal value.
        a_high = a_low
    t_start_a = int(rng.integers(a_low, a_high + 1))
    t_start_b = t_start_a + lag
    t_end_a = t_start_a + width
    t_end_b = t_start_b + width
    target_a = _pick_target(scene_a, rng)
    target_b = _pick_target(scene_b, rng)
    magnitude = float(rng.uniform(*difficulty.magnitude_range))
    spec_a = InjectionSpec(channel=target_a, t_start=t_start_a, t_end=t_end_a,
                           kind=anomaly_kind, magnitude=magnitude)
    spec_b = InjectionSpec(channel=target_b, t_start=t_start_b, t_end=t_end_b,
                           kind=anomaly_kind, magnitude=magnitude)
    scene_a.channels[target_a] = inject(scene_a.channels[target_a], spec_a, rng)
    scene_b.channels[target_b] = inject(scene_b.channels[target_b], spec_b, rng)
    injections.extend([spec_a, spec_b])
    return injections, coupling


def merge_series(scenes: Iterable[Scene]) -> dict[str, list[float]]:
    """Flatten two scenes' channels into a single name->list dict (float32-ish)."""
    out: dict[str, list[float]] = {}
    for sc in scenes:
        for name, arr in sc.channels.items():
            out[name] = arr.astype(np.float32).tolist()
    return out
