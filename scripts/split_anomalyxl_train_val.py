"""Deterministic 90/10 train/val split for AnomalyXL training shards.

Splits ``data.parquet`` in a shard directory by hashing each row's
source precise ``id`` (mod 10): partitions 0..8 -> train, 9 -> val.

For paired shards the row id is ``paired__<src_id>__<template>``; we
strip the ``paired__`` prefix and the trailing ``__<template>`` so a
precise row and its projected coarse rows land in the same partition.
That keeps any future pair-wise analysis intact.

Usage:
    uv run python scripts/split_anomalyxl_train_val.py \\
        --src anomalyXL/datasets/anomalyxl-train/anomalyxl-precise

Writes ``train.parquet`` + ``val.parquet`` next to ``data.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


_PAIRED_PREFIX = "paired__"


def _precise_id_of(row_id: str) -> str:
    if row_id.startswith(_PAIRED_PREFIX):
        stripped = row_id[len(_PAIRED_PREFIX):]
        # paired__<src_id>__<template> — strip trailing __<template>
        if "__" in stripped:
            return stripped.rsplit("__", 1)[0]
        return stripped
    return row_id


def _partition(row_id: str) -> int:
    src = _precise_id_of(row_id)
    h = hashlib.blake2b(src.encode("utf-8"), digest_size=4).hexdigest()
    return int(h, 16) % 10


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True,
                   help="Shard dir containing data.parquet.")
    p.add_argument("--val-bucket", type=int, default=9,
                   help="Hash bucket (0..9) routed to val. Default 9.")
    args = p.parse_args()

    src_parquet = args.src / "data.parquet"
    if not src_parquet.exists():
        raise SystemExit(f"No data.parquet at {args.src}")

    table = pq.read_table(src_parquet)
    ids = table.column("id").to_pylist()
    buckets = [_partition(rid) for rid in ids]

    bucket_array = pa.array(buckets, type=pa.int8())
    val_mask = pc.equal(bucket_array, args.val_bucket)
    train_mask = pc.invert(val_mask)

    train_table = table.filter(train_mask)
    val_table = table.filter(val_mask)

    pq.write_table(train_table, args.src / "train.parquet")
    pq.write_table(val_table, args.src / "val.parquet")

    print(f"  train: {train_table.num_rows:5d}  -> {args.src / 'train.parquet'}")
    print(f"  val:   {val_table.num_rows:5d}  -> {args.src / 'val.parquet'}")


if __name__ == "__main__":
    main()
