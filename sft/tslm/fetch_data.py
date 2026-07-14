#!/usr/bin/env python
"""Materialize the AnomalyXL shards the TSLM configs read, from the released Hub repo.

The configs expect, relative to the repo root:

  anomalyXL/datasets/anomalyxl-train/anomalyxl-{precise,coarse}/train.parquet
  anomalyXL/datasets/anomalyxl-train/anomalyxl-{precise,coarse}/val.parquet
  anomalyXL/datasets/anomalyxl-{precise,coarse}/data.parquet          (gold test)

which map onto the `nz00shuuuu/AnomalyXL` monorepo partitions: precise -> `Localize`,
coarse -> `MCQ`, each shipping `{train,val,test}.parquet`.

  uv run python fetch_data.py                 # both variants
  uv run python fetch_data.py --variant precise
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO = "nz00shuuuu/AnomalyXL"
PARTITIONS = {"precise": "Localize", "coarse": "MCQ"}
SPLITS = {"train": "train.parquet", "val": "val.parquet"}


def fetch(variant: str) -> None:
    partition = PARTITIONS[variant]
    train_dir = REPO_ROOT / "anomalyXL/datasets/anomalyxl-train" / f"anomalyxl-{variant}"
    test_dir = REPO_ROOT / "anomalyXL/datasets" / f"anomalyxl-{variant}"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for split, local_name in SPLITS.items():
        src = hf_hub_download(REPO, f"{partition}/{split}.parquet", repo_type="dataset")
        shutil.copyfile(src, train_dir / local_name)
        print(f"  {REPO}:{partition}/{split:5s} -> {train_dir / local_name}")

    src = hf_hub_download(REPO, f"{partition}/test.parquet", repo_type="dataset")
    shutil.copyfile(src, test_dir / "data.parquet")
    print(f"  {REPO}:{partition}/{'test':5s} -> {test_dir / 'data.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["precise", "coarse", "both"], default="both")
    args = ap.parse_args()
    variants = ["precise", "coarse"] if args.variant == "both" else [args.variant]
    for v in variants:
        fetch(v)


if __name__ == "__main__":
    main()
