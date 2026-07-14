"""Shared bucketing logic for Start Time and End Time templates.

Both pick an ordinal label first, then derive a concrete index range from the
actual `L`. Option text bakes the index range in so the gold is
self-describing (e.g. `"Early (indices 205-409)"`).
"""

from __future__ import annotations

import numpy as np

__all__ = ["bucket_bounds", "bucket_label", "QUARTER_LABELS_START", "QUARTER_LABELS_END"]


# Four equal-width buckets covering [0, L).
QUARTER_LABELS_START: tuple[str, ...] = ("Before", "Early", "Medium", "Late")
QUARTER_LABELS_END: tuple[str, ...] = ("Early", "Medium", "Late", "Latest")


def bucket_bounds(L: int, bucket: int, n_buckets: int = 4) -> tuple[int, int]:
    """Inclusive-exclusive bounds for bucket `bucket` of `n_buckets` over [0, L)."""
    if not 0 <= bucket < n_buckets:
        raise ValueError(f"bucket {bucket} outside [0, {n_buckets})")
    lo = (L * bucket) // n_buckets
    hi = (L * (bucket + 1)) // n_buckets
    return lo, hi


def bucket_label(prefix: str, L: int, bucket: int, n_buckets: int = 4) -> str:
    """`"Early (indices 205-409)"` — bake the range into the option text."""
    lo, hi = bucket_bounds(L, bucket, n_buckets)
    return f"{prefix} (indices {lo}-{hi - 1})"


def index_to_bucket(idx: int, L: int, n_buckets: int = 4) -> int:
    """Inverse of `bucket_bounds`. Clamps to last bucket if `idx == L`."""
    if idx >= L:
        return n_buckets - 1
    return min(n_buckets - 1, (idx * n_buckets) // L)
