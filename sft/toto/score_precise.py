#!/usr/bin/env python
"""Score an AnomalyXL-precise eval_toto_vlm.py results CSV with the real
AnomalyXL precise scorer (IoU / kind-match / composite per category).

eval_toto_vlm.py's own `is_correct` column is raw string equality against the
full `category:{json}` gold string — the right metric for the `coarse`
(MCQ) variant, but not for `precise`: the gold there is a JSON payload with
exact integer endpoints, so `is_correct` is ~always False regardless of how
close the model's answer actually is. This script re-scores the `precise`
rows properly, using the same scorer TimeRLM/the TSLM baselines/the classical
baseline all use (`src/utils/anomalyxl_scoring.py`, vendored from
`../tslm/src/datasets/anomaly_xl/scoring.py`), against `prediction_raw` (the
model's undecoded generation, saved by eval.py alongside the parsed
`model_answer` column).

Usage:
  uv run python score_precise.py results/toto_ft_precise_eval_results.csv \
      --out results/toto_ft_precise_scored.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils.anomalyxl_scoring import score_precise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv", help="a *_results.csv written by eval.py")
    ap.add_argument("--out", required=True, help="per-row scored CSV to write")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)

    rows = []
    by_category = defaultdict(list)
    for _, r in df.iterrows():
        pred = r["prediction_raw"] if pd.notna(r["prediction_raw"]) else ""
        result = score_precise(str(pred), str(r["correct_answer"]))
        rows.append({
            "query_group": r["query_group"],
            "task_category": r["task_category"],
            "correct_answer": r["correct_answer"],
            "prediction_raw": pred,
            **result,
        })
        by_category[result["category"]].append(result["primary"])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print(f"scored {len(rows)} rows -> {args.out}")
    print(f"overall primary mean: {out_df['primary'].mean():.4f}")
    for cat, scores in sorted(by_category.items()):
        print(f"  {cat}: {sum(scores) / len(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
