"""`python -m anomalyxl generate ...` — build a parquet shard from a grid spec.

Writes one parquet file and a sibling `manifest.json`. The parquet uses
parallel-array `channel_names`/`channel_values` columns so variable channel
counts coexist in a single Arrow schema (see `PLAN.md §7`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .pipeline import SampleSpec, generate, row_id, specs_from_grid
from .questions import TEMPLATES

# Channel values are rendered to the model via `str(v)`. Storing at float32
# precision (e.g. 0.123 -> 0.12300000339746475) defeats compactness, and
# anomalyxl scenes don't need more than ~3 decimals (noise + structure are
# O(1); anomalies up to O(100)). Round at write time for better parquet
# compression; the read path applies the same rounding to clean up the
# float32 -> float64 round-trip noise in `str()`.
_VALUE_DECIMALS = 3


_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("tier", pa.int8()),
    ("category", pa.string()),
    ("length", pa.int32()),
    ("n_channels", pa.int32()),
    ("n_channels_a", pa.int32()),
    ("n_channels_b", pa.int32()),
    ("question", pa.string()),
    ("options", pa.list_(pa.string())),
    ("answer", pa.string()),
    ("semantic_class", pa.string()),
    ("coupling_kind", pa.string()),
    ("coupling_lag", pa.int32()),
    ("injection_kinds", pa.list_(pa.string())),
    ("injection_channels", pa.list_(pa.string())),
    ("injection_starts", pa.list_(pa.int32())),
    ("injection_ends", pa.list_(pa.int32())),
    ("channel_names", pa.list_(pa.string())),
    ("channel_values", pa.list_(pa.list_(pa.float32()))),
    ("seed", pa.int64()),
])


def _row_to_record(spec: SampleSpec, row) -> dict:
    paired = spec.paired_split is not None
    template = TEMPLATES[spec.category]
    rec = {
        "id": row_id(spec),
        "tier": int(template.tier),
        "category": spec.category,
        "length": int(spec.length),
        "n_channels": int(len(row.series)),
        "n_channels_a": None,
        "n_channels_b": None,
        "question": row.question,
        "options": list(row.options),
        "answer": row.answer,
        "semantic_class": row.semantic_class,
        "coupling_kind": row.coupling.kind if row.coupling else None,
        "coupling_lag": int(row.coupling.lag) if row.coupling else None,
        "injection_kinds": [inj.kind for inj in row.injections],
        "injection_channels": [inj.channel for inj in row.injections],
        "injection_starts": [int(inj.t_start) for inj in row.injections],
        "injection_ends": [int(inj.t_end) for inj in row.injections],
        "channel_names": list(row.series.keys()),
        "channel_values": [
            np.round(np.asarray(v, dtype=np.float32), _VALUE_DECIMALS).tolist()
            for v in row.series.values()
        ],
        "seed": int(spec.seed),
    }
    if paired:
        a = sum(1 for n in row.series if n.startswith("A_"))
        b = sum(1 for n in row.series if n.startswith("B_"))
        rec["n_channels_a"] = int(a)
        rec["n_channels_b"] = int(b)
    return rec


def _resolve_categories(arg: list[str]) -> list[str]:
    if arg == ["all"]:
        return sorted(TEMPLATES.keys())
    unknown = [c for c in arg if c not in TEMPLATES]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}. Available: {sorted(TEMPLATES)}")
    return list(arg)


def _generate_cmd(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = _resolve_categories(args.categories)
    specs = specs_from_grid(
        lengths=args.lengths,
        channels=args.channels,
        per_cell=args.per_cell,
        categories=categories,
        seed=args.seed,
    )
    parquet_path = out_dir / "data.parquet"
    writer = pq.ParquetWriter(parquet_path, _SCHEMA)
    written = 0
    try:
        buffer: list[dict] = []
        flush_every = max(1, args.flush_every)
        for spec, row in generate(specs):
            buffer.append(_row_to_record(spec, row))
            if len(buffer) >= flush_every:
                writer.write_table(pa.Table.from_pylist(buffer, schema=_SCHEMA))
                written += len(buffer)
                buffer = []
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=_SCHEMA))
            written += len(buffer)
    finally:
        writer.close()
    manifest = {
        "version": "0.1.0",
        "seed": args.seed,
        "lengths": list(args.lengths),
        "channels": list(args.channels),
        "categories": categories,
        "per_cell": args.per_cell,
        "n_rows": written,
        "schema": [name for name in _SCHEMA.names],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {written} rows to {parquet_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="anomalyxl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a parquet shard from a grid spec.")
    g.add_argument("--out", required=True, type=str)
    g.add_argument("--lengths", nargs="+", type=int, required=True)
    g.add_argument("--channels", nargs="+", type=int, required=True)
    g.add_argument("--per-cell", type=int, default=25)
    g.add_argument("--categories", nargs="+", default=["all"])
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--flush-every", type=int, default=64,
                   help="Write a parquet row group every N rows (memory cap).")
    g.set_defaults(func=_generate_cmd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
