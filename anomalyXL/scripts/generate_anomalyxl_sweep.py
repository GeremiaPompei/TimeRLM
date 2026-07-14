"""Generate both AnomalyXL release shards in one shot.

Emits two parquet datasets per `AnomalyXL_PLAN.md` §5:

- ``anomalyxl-coarse/``: 250 rows of v1 MQA categories at L=16384 (the
  ARFBench-equivalent length). The "coarse" precursor — quarter-bucket
  start times, 4-bucket magnitudes, kind picker, etc. Used as the
  baseline so the L=16k drop between coarse MQA scoring and precise
  evidence-grounded scoring attributes the headline collapse to
  "stricter scoring" rather than "longer context".
- ``anomalyxl-precise/``: ~800 rows of the 5 evidence-grounded
  categories across the length grid (exact sample-index gold,
  exact σ-units, exact event sets). The headline benchmark.

Each shard ships ``data.parquet`` + ``manifest.json``; the manifest
``artifact`` field is ``"coarse"`` or ``"precise"`` so downstream
consumers can route per release.

Pattern lifted from ``generate_pilot.py`` — each sub-grid calls
``anomalyxl.cli._generate_cmd`` into a temp directory, then ``pyarrow``
concatenates the per-grid parquets into one shard parquet.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from anomalyxl.cli import _SCHEMA, _generate_cmd

_DEFAULT_SEED: int = 42
_COARSE_PER_CELL: int = 50
_PRECISE_PER_CELL: int = 50


@dataclass(frozen=True)
class _SubGrid:
    """One (channels × lengths × categories × per_cell) call to the CLI."""
    label: str
    channels: tuple[int, ...]
    lengths: tuple[int, ...]
    categories: tuple[str, ...]
    per_cell: int
    seed_offset: int  # added to global seed so sub-grids don't share an rng


# Coarse (precursor): 5 v1 MQA categories @ L=16k. Split into two sub-grids
# because `identification` needs C=4 while the other four are C=1.
_COARSE_GRIDS: tuple[_SubGrid, ...] = (
    _SubGrid(
        label="coarse_C1",
        channels=(1,),
        lengths=(16384,),
        categories=("presence", "start_time", "categorization", "magnitude"),
        per_cell=_COARSE_PER_CELL,
        seed_offset=0,
    ),
    _SubGrid(
        label="coarse_C4",
        channels=(4,),
        lengths=(16384,),
        categories=("identification",),
        per_cell=_COARSE_PER_CELL,
        seed_offset=1,
    ),
)


# Precise (main): 5 evidence categories. Three sub-grids by channel topology.
_PRECISE_GRIDS: tuple[_SubGrid, ...] = (
    _SubGrid(
        label="precise_C1",
        channels=(1,),
        lengths=(16384, 32768, 65536, 131072),
        categories=("localize", "classify_with_evidence", "measure_magnitude"),
        per_cell=_PRECISE_PER_CELL,
        seed_offset=10,
    ),
    _SubGrid(
        label="precise_C16",
        channels=(16,),
        lengths=(32768, 131072),
        categories=("localize_all_channels",),
        per_cell=_PRECISE_PER_CELL,
        seed_offset=11,
    ),
    _SubGrid(
        label="precise_C2_paired",
        channels=(2,),
        lengths=(32768, 131072),
        categories=("lead_lag_with_magnitude",),
        per_cell=_PRECISE_PER_CELL,
        seed_offset=12,
    ),
)


def _run_subgrid(tmp_dir: Path, grid: _SubGrid, global_seed: int) -> Path:
    args = argparse.Namespace(
        out=str(tmp_dir),
        lengths=list(grid.lengths),
        channels=list(grid.channels),
        per_cell=grid.per_cell,
        categories=list(grid.categories),
        seed=global_seed + grid.seed_offset,
        flush_every=64,
    )
    _generate_cmd(args)
    return tmp_dir / "data.parquet"


def _concat_shards(out_dir: Path, sub_parquets: list[Path]) -> int:
    tables = [pq.read_table(p) for p in sub_parquets]
    combined = pa.concat_tables(tables)
    pq.write_table(combined, out_dir / "data.parquet")
    return combined.num_rows


def _write_manifest(out_dir: Path, artifact: str, grids: tuple[_SubGrid, ...],
                    seed: int, n_rows: int) -> None:
    manifest = {
        "version": "0.2.0",
        "artifact": artifact,
        "seed": seed,
        "grids": [
            {
                "label": g.label,
                "channels": list(g.channels),
                "lengths": list(g.lengths),
                "categories": list(g.categories),
                "per_cell": g.per_cell,
                "seed_offset": g.seed_offset,
            }
            for g in grids
        ],
        "n_rows": n_rows,
        "schema": list(_SCHEMA.names),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _generate_shard(
    out_dir: Path,
    artifact: str,
    grids: tuple[_SubGrid, ...],
    seed: int,
    per_cell_override: int | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / "_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir()

    sub_parquets: list[Path] = []
    effective_grids: list[_SubGrid] = []
    for grid in grids:
        if per_cell_override is not None:
            grid = _SubGrid(
                label=grid.label,
                channels=grid.channels,
                lengths=grid.lengths,
                categories=grid.categories,
                per_cell=per_cell_override,
                seed_offset=grid.seed_offset,
            )
        effective_grids.append(grid)
        sub_dir = tmp_root / grid.label
        sub_dir.mkdir()
        print(f"  [{artifact}] {grid.label}: channels={grid.channels} "
              f"lengths={grid.lengths} cats={grid.categories} per_cell={grid.per_cell}")
        sub_parquets.append(_run_subgrid(sub_dir, grid, seed))

    n_rows = _concat_shards(out_dir, sub_parquets)
    _write_manifest(out_dir, artifact, tuple(effective_grids), seed, n_rows)
    shutil.rmtree(tmp_root)
    print(f"  -> {artifact}: {n_rows} rows -> {out_dir / 'data.parquet'}")
    return n_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the anomalyxl-coarse + anomalyxl-precise release shards."
    )
    p.add_argument("--out-root", required=True, type=Path,
                   help="Parent dir; emits {out-root}/anomalyxl-coarse and {out-root}/anomalyxl-precise.")
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--per-cell", type=int, default=None,
                   help="Override per-cell row count for both shards (smoke runs).")
    p.add_argument("--coarse-only", action="store_true",
                   help="Generate only the precursor (coarse MQA) shard.")
    p.add_argument("--precise-only", action="store_true",
                   help="Generate only the headline (precise evidence-grounded) shard.")
    args = p.parse_args()

    if args.coarse_only and args.precise_only:
        raise SystemExit("--coarse-only and --precise-only are mutually exclusive.")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    if not args.precise_only:
        total_rows += _generate_shard(
            out_root / "anomalyxl-coarse", "coarse", _COARSE_GRIDS, args.seed,
            per_cell_override=args.per_cell,
        )
    if not args.coarse_only:
        total_rows += _generate_shard(
            out_root / "anomalyxl-precise", "precise", _PRECISE_GRIDS, args.seed,
            per_cell_override=args.per_cell,
        )

    print(f"\nDone. {total_rows} rows total across both shards.")


if __name__ == "__main__":
    main()
