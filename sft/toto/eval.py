#!/usr/bin/env python
"""Evaluate Toto-1.0-QA-Experimental on an AnomalyXL export — zero-shot
(base model) or with a fine-tuned LoRA adapter, on the same input pipeline,
so the two are directly comparable.

Usage:
  # Zero-shot (base model, no fine-tuning):
  CUDA_VISIBLE_DEVICES=<gpu> uv run python eval.py \\
      --data ../../anomalyXL/datasets/toto-export/coarse \\
      --output-prefix results/toto_zeroshot_coarse

  # Fine-tuned (merges the LoRA adapter into the VLM before running inference):
  CUDA_VISIBLE_DEVICES=<gpu> uv run python eval.py \\
      --data ../../anomalyXL/datasets/toto-export/coarse \\
      --adapter checkpoints/toto_ft_coarse/adapter \\
      --output-prefix results/toto_ft_coarse_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation.eval_toto_vlm import (  # noqa: E402
    evaluate_toto_vlm_model,
    load_model_and_processor,
)
from utils.log_utils import save_results_and_statistics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter dir (e.g. checkpoints/toto_ft_coarse/adapter); "
                         "omit for zero-shot base-model evaluation")
    ap.add_argument("--data", required=True,
                    help="export dir with benchmark.csv + arfbench-ts-data + arfbench-images")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--model-path", default="Datadog/Toto-1.0-QA-Experimental")
    ap.add_argument("--max-ts-length", type=int, default=12800)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (smoke test)")
    args = ap.parse_args()

    model, processor = load_model_and_processor(args.model_path)

    if args.adapter:
        # Apply the fine-tuned LoRA on the text decoder and merge for plain inference.
        from peft import PeftModel
        model.vlm = PeftModel.from_pretrained(model.vlm, args.adapter)
        model.vlm = model.vlm.merge_and_unload()
    model.eval()

    data = Path(args.data)
    benchmark_df = pd.read_csv(data / "benchmark.csv")
    if args.limit:
        benchmark_df = benchmark_df.head(args.limit)
    results_df = evaluate_toto_vlm_model(
        model=model,
        processor=processor,
        benchmark_df=benchmark_df,
        image_dir=str(data / "arfbench-images"),
        use_timeseries=True,
        data_dir=str(data / "arfbench-ts-data"),
        ts_interval=0,
        max_ts_length=args.max_ts_length,
        short_max_ts_groups=100,
        long_max_ts_groups=1000,
        temperature=0.0,
        max_new_tokens=2000,
        batch_size=1,
        output_file=f"{args.output_prefix}_results.csv",
        resume=True,
    )
    save_results_and_statistics(results_df, args.output_prefix)


if __name__ == "__main__":
    main()
