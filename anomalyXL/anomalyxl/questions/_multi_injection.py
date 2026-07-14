"""Helper for AnomalyXL multi-channel templates: place K non-overlapping
anomaly windows across K distinct channels of a scene.

Used by `localize_all_channels` (K ∈ {0..3} per row, channels picked
uniformly from the scene's pool). Windows on *different* channels can
overlap in time — that's not a collision for set-matching purposes (the
scorer requires both channel and window to match). Same-channel
overlap is rejected when `same_channel_allowed=True` (off by default,
since v2 has no use case for it yet).
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, InjectionSpec, make_injection, inject
from ..difficulty import Difficulty
from ..scene import Scene

__all__ = ["place_k_anomalies"]


def place_k_anomalies(
    scene: Scene,
    k: int,
    rng: np.random.Generator,
    *,
    same_channel_allowed: bool = False,
    max_attempts: int = 32,
    difficulty: "Difficulty | None" = None,
) -> list[InjectionSpec]:
    """Place `k` anomalies in `scene`, each on a distinct channel.

    Returns the list of placed `InjectionSpec`s in the order they were
    injected (which is also the order channels were sampled). Mutates
    `scene.channels` in place — caller owns the scene.

    `k = 0` is a first-class case: returns `[]` without touching the
    scene. `k` must be `<= len(scene.names)` (or `<= len(scene.names) *
    max_attempts` when `same_channel_allowed=True`, but that path
    isn't exercised by v2).
    """
    difficulty = difficulty or Difficulty()
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return []
    if not same_channel_allowed and k > len(scene.names):
        raise ValueError(
            f"k={k} > n_channels={len(scene.names)} with distinct-channel "
            "constraint; relax same_channel_allowed or shrink k."
        )

    L = scene.length
    names = list(scene.names)
    if same_channel_allowed:
        chosen_channels: list[str] = [
            names[int(rng.integers(0, len(names)))] for _ in range(k)
        ]
    else:
        idx = rng.choice(len(names), size=k, replace=False)
        chosen_channels = [names[int(i)] for i in idx]

    placed: list[InjectionSpec] = []
    per_channel_windows: dict[str, list[tuple[int, int]]] = {}

    for channel in chosen_channels:
        kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        spec: InjectionSpec | None = None
        for _ in range(max_attempts):
            candidate = make_injection(
                L, rng, kind=kind, channel=channel,
                min_width=difficulty.width_range[0], max_width=difficulty.width_range[1],
                magnitude_range=difficulty.magnitude_range,
            )
            existing = per_channel_windows.get(channel, [])
            if not _overlaps_any(candidate.t_start, candidate.t_end, existing):
                spec = candidate
                break
        if spec is None:
            # `make_injection` window distribution is uniform over a wide
            # range; for v2's k<=3 and absolute width <=256 the rejection
            # rate is negligible. Hitting `max_attempts` means the channel
            # is already saturated — surface it rather than silently
            # double-stack.
            raise RuntimeError(
                f"Could not place anomaly on channel {channel!r} within "
                f"{max_attempts} attempts (k={k}, L={L})."
            )

        scene.channels[channel] = inject(scene.channels[channel], spec, rng)
        per_channel_windows.setdefault(channel, []).append((spec.t_start, spec.t_end))
        placed.append(spec)

    return placed


def _overlaps_any(start: int, end: int, windows: list[tuple[int, int]]) -> bool:
    """Half-open interval overlap check against a list of existing windows."""
    for s, e in windows:
        if start < e and s < end:
            return True
    return False
