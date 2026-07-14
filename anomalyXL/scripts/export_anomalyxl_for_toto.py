#!/usr/bin/env python
"""Export an AnomalyXL dataset to the ARFBench eval layout consumed by
``toto-1.0-qa-experimental/evaluation/eval_toto_vlm.py``: a benchmark CSV plus
``arfbench-ts-data/{id}_{interval}.parquet`` and ``arfbench-images/{id}.png``.

Series are written on the ARFBench interval grid (10/60/300/1800s) by strided
decimation so the eval's finest-that-fits auto-selection works unchanged.
Timestamps are synthetic (10s per index from a fixed origin) — AnomalyXL
questions and answers reference *indices*, so the plots use an index x-axis
and the timestamps only serve Toto's encoder, which is shape-based.

Usage:
  .venv/bin/python scripts/export_anomalyxl_for_toto.py \
      datasets/anomalyxl-coarse out/toto-anomalyxl-coarse
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ARFBench interval suffix -> decimation stride (base grid is 10s per index).
STRIDES = {10: 1, 60: 6, 300: 30, 1800: 180}
ORIGIN = pd.Timestamp("2025-01-01")
PLOT_MAX_POINTS = 4000


def export(src: str, out: str, parquet: str = "data.parquet") -> None:
    df = pd.read_parquet(Path(src) / parquet)
    out_p = Path(out)
    ts_dir = out_p / "arfbench-ts-data"
    img_dir = out_p / "arfbench-images"
    ts_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        qg = r["id"]
        options = [str(o) for o in r["options"]]
        rows.append({
            "question": r["question"],
            "task_category": r["category"],
            "difficulty": f"Tier {r['tier']}",
            "options_str": json.dumps(options),
            "correct_answer": str(r["answer"]),
            "query_group": qg,
            "options": json.dumps(options),
            "interpolate_1": 0,
            "interpolate_2": 0,
        })

        chans = [str(c) for c in r["channel_names"]]
        vals = [np.asarray(v, dtype=np.float32) for v in r["channel_values"]]
        length = len(vals[0])
        for interval, stride in STRIDES.items():
            idx = np.arange(0, length, stride)
            epoch = ORIGIN + pd.to_timedelta(idx * 10, unit="s")
            pd.concat(
                [pd.DataFrame({"epoch": epoch, "group": ch, "value": v[idx]})
                 for ch, v in zip(chans, vals)],
                ignore_index=True,
            ).to_parquet(ts_dir / f"{qg}_{interval}.parquet", index=False)

        step = max(1, length // PLOT_MAX_POINTS)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        for ch, v in zip(chans, vals):
            ax.plot(np.arange(0, length, step), v[::step], linewidth=0.8,
                    label=ch if len(chans) <= 8 else None)
        if len(chans) <= 8:
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("index")
        fig.tight_layout()
        fig.savefig(img_dir / f"{qg}.png")
        plt.close(fig)
        if n % 100 == 0:
            print(f"  {n}/{len(df)}", flush=True)

    pd.DataFrame(rows).to_csv(out_p / "benchmark.csv", index=False)
    print(f"exported {len(rows)} rows -> {out_p}", flush=True)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
