from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from anomalyxl.cli import main
from anomalyxl.pipeline import generate, row_id, specs_from_grid


def test_specs_from_grid_shape_and_uniqueness():
    specs = specs_from_grid(
        lengths=[64, 128],
        channels=[1, 2],
        per_cell=3,
        categories=["categorization"],
        seed=42,
    )
    assert len(specs) == 2 * 2 * 1 * 3
    # All seeds are distinct -> deterministic per-cell ordering.
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == len(seeds)


def test_specs_from_grid_seed_stability_across_extra_per_cell():
    a = specs_from_grid(lengths=[64], channels=[1], per_cell=2, categories=["categorization"], seed=42)
    b = specs_from_grid(lengths=[64], channels=[1], per_cell=4, categories=["categorization"], seed=42)
    # The first per_cell rows in `b` must match `a` (so reruns can grow per_cell
    # without reshuffling prior data).
    assert [s.seed for s in a] == [s.seed for s in b][:2]


def test_generate_yields_categorization_rows():
    specs = specs_from_grid([64], [1], per_cell=2, categories=["categorization"], seed=0)
    out = list(generate(specs))
    assert len(out) == 2
    for spec, row in out:
        assert spec.category == "categorization"
        assert row.answer in row.options


def test_row_id_is_stable():
    specs = specs_from_grid([64], [1], 1, ["categorization"], seed=0)
    assert row_id(specs[0]) == row_id(specs[0])


def test_cli_writes_parquet_and_manifest(tmp_path: Path):
    out = tmp_path / "shard"
    main([
        "generate",
        "--out", str(out),
        "--lengths", "64", "128",
        "--channels", "1", "2",
        "--per-cell", "2",
        "--categories", "categorization",
        "--seed", "7",
    ])
    assert (out / "data.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_rows"] == 2 * 2 * 1 * 2
    table = pq.read_table(out / "data.parquet")
    assert table.num_rows == manifest["n_rows"]
    df = table.to_pandas()
    # Spot-check schema fields the env will read.
    for col in ("id", "category", "length", "n_channels", "question", "options",
                "answer", "semantic_class", "channel_names", "channel_values"):
        assert col in df.columns
    # Every row's gold appears in options.
    for _, row in df.iterrows():
        assert row["answer"] in list(row["options"])
    # Series length matches the configured length.
    for _, row in df.iterrows():
        for values in row["channel_values"]:
            assert len(values) == int(row["length"])
