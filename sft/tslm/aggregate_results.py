"""Aggregate AnomalyXL TSLM eval results across the 6 (model × variant) runs.

Reads each run's ``output_logs/test_epoch_*.json`` (the validate() loop's
log file) under ``results/<run_name>/`` and emits a single comparison
table: per-question-category primary metric for each model × variant,
plus an aggregate row. Optionally cross-references against the
GPT-5.5 / Qwen3.5-4B baselines if their merged-result JSONLs are
findable.

Usage:
    uv run python scripts/aggregate_tslm_results.py \\
        --results-dir results \\
        --out anomaly_xl_tslm_comparison.tsv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


_RUN_NAMES = [
    ("flamingo", "precise", "anomaly_xl_flamingo_precise_chronos2_qwen35_4b"),
    ("flamingo", "coarse",  "anomaly_xl_flamingo_coarse_chronos2_qwen35_4b"),
    ("chatts",   "precise", "anomaly_xl_chatts_precise_chronos2_qwen35_4b"),
    ("chatts",   "coarse",  "anomaly_xl_chatts_coarse_chronos2_qwen35_4b"),
    ("itformer", "precise", "anomaly_xl_itformer_precise_chronos2_qwen35_4b"),
    ("itformer", "coarse",  "anomaly_xl_itformer_coarse_chronos2_qwen35_4b"),
]


def _latest_test_log(run_dir: Path) -> Path | None:
    logs_dir = run_dir / "output_logs"
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("test_epoch_*.json"))
    return candidates[-1] if candidates else None


def _per_category(entries: list[dict]) -> dict[str, dict[str, float]]:
    """Bucket entries by category, compute count + mean(primary) + mean(correct)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_cat[e.get("category", "unknown")].append(e)
    out: dict[str, dict[str, float]] = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        primary_sum = sum(float(r.get("primary", 0.0)) for r in rows)
        correct_sum = sum(1 for r in rows if r.get("correct"))
        out[cat] = {
            "n": n,
            "primary_mean": primary_sum / max(n, 1),
            "exact_acc": correct_sum / max(n, 1),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("anomaly_xl_tslm_comparison.tsv"))
    args = p.parse_args()

    rows = []
    all_categories: set[str] = set()
    per_run: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    for model, variant, run_name in _RUN_NAMES:
        run_dir = args.results_dir / run_name
        log_path = _latest_test_log(run_dir)
        if log_path is None:
            print(f"  [SKIP] {model}/{variant}: no test_epoch_*.json under {run_dir}")
            per_run[(model, variant)] = {}
            continue
        entries = json.loads(log_path.read_text())
        cats = _per_category(entries)
        per_run[(model, variant)] = cats
        all_categories.update(cats)
        print(f"  [OK]   {model}/{variant}: {len(entries)} rows, {len(cats)} categories ({log_path.name})")

    categories = sorted(all_categories)
    header = ["model", "variant"] + [f"{c}_primary" for c in categories] + [f"{c}_n" for c in categories] + ["overall_primary"]
    lines = ["\t".join(header)]
    for model, variant, _ in _RUN_NAMES:
        cats = per_run[(model, variant)]
        primary_cells = [f"{cats.get(c, {}).get('primary_mean', 0.0):.4f}" if c in cats else "" for c in categories]
        n_cells = [str(cats.get(c, {}).get("n", "")) for c in categories]
        total_n = sum(cats.get(c, {}).get("n", 0) for c in categories)
        overall = (
            sum(cats.get(c, {}).get("primary_mean", 0.0) * cats.get(c, {}).get("n", 0) for c in categories) / total_n
            if total_n else 0.0
        )
        lines.append("\t".join([model, variant] + primary_cells + n_cells + [f"{overall:.4f}"]))

    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out} ({len(lines)-1} runs × {len(categories)} categories)")


if __name__ == "__main__":
    main()
