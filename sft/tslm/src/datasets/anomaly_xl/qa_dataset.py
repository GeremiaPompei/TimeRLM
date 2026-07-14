"""AnomalyXL QADataset.

Two registry names (``anomaly_xl_precise``, ``anomaly_xl_coarse``) both
point at this class; the ``variant`` kwarg (set via
``dataset.extra_kwargs.variant`` in the config) selects answer format
and gold-test directory.

Train + val are read from ``train_dir/{train,val}.parquet`` (produced
by ``scripts/split_anomalyxl_train_val.py``). Test is
``test_dir/data.parquet`` (defaults to the gold seed=42 release shards
in ``anomalyXL/datasets/anomalyxl-{precise,coarse}/``).

``train_dir``/``test_dir`` may be given as paths relative to the
repo root (this keeps configs portable across machines/
clones); absolute paths are also accepted unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset

from src.datasets.anomaly_xl.scoring import (
    COARSE_SYSTEM_PROMPT,
    PRECISE_SYSTEM_PROMPT,
    PRIMARY_METRIC,
    is_precise_gold,
    score_any,
)
from src.datasets.qa_base import QADataset
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt


# src/datasets/anomaly_xl/qa_dataset.py -> src -> tslm -> sft -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]

_GOLD_TEST_DIR = {
    "precise": "anomalyXL/datasets/anomalyxl-precise",
    "coarse": "anomalyXL/datasets/anomalyxl-coarse",
}


def _resolve_repo_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else _REPO_ROOT / p)


def _load_parquet_as_hf(path: Path) -> Dataset:
    table = pq.read_table(path)
    return Dataset(table)


class AnomalyXLDataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        variant: Literal["precise", "coarse"] = "precise",
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
    ):
        if variant not in ("precise", "coarse"):
            raise ValueError(f"variant must be 'precise' or 'coarse', got {variant!r}")
        if train_dir is None:
            raise ValueError(
                "AnomalyXLDataset requires train_dir (path to dir containing "
                "train.parquet + val.parquet). Set dataset.extra_kwargs.train_dir "
                "in the config."
            )
        self.variant = variant
        self.train_dir = _resolve_repo_path(train_dir)
        self.test_dir = _resolve_repo_path(test_dir if test_dir else _GOLD_TEST_DIR[variant])

        super().__init__(
            split,
            EOS_TOKEN,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
            lazy_loading=lazy_loading,
        )

    def _dataset_cache_key(self) -> tuple[Any, ...]:
        return (self.variant, self.train_dir, self.test_dir)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_root = Path(self.train_dir)
        test_root = Path(self.test_dir)
        train_path = train_root / "train.parquet"
        val_path = train_root / "val.parquet"
        test_path = test_root / "data.parquet"
        for label, p in (("train", train_path), ("val", val_path), ("test", test_path)):
            if not p.exists():
                raise FileNotFoundError(f"Missing {label} parquet at {p}")
        return (
            _load_parquet_as_hf(train_path),
            _load_parquet_as_hf(val_path),
            _load_parquet_as_hf(test_path),
        )

    def _get_answer(self, row) -> str:
        return str(row["answer"])

    def _get_pre_prompt(self, row) -> str:
        if self.variant == "precise":
            return PRECISE_SYSTEM_PROMPT
        return COARSE_SYSTEM_PROMPT

    def _get_post_prompt(self, row) -> str:
        question = row["question"]
        if self.variant == "precise":
            return f"Question: {question}"
        options = row.get("options") or []
        if options:
            import json as _json
            opts_str = _json.dumps(list(options), ensure_ascii=False)
            return f"Question: {question}\nOptions: {opts_str}"
        return f"Question: {question}"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        names = list(row["channel_names"])
        values = row["channel_values"]
        prompts: List[TextTimeSeriesPrompt] = []
        for name, channel in zip(names, values):
            arr = np.asarray(channel, dtype=np.float32)
            prompts.append(
                TextTimeSeriesPrompt(f"The following is channel {name}:", arr)
            )
        return prompts

    @property
    def category_key(self) -> str:
        return "category"

    @property
    def primary_metric(self) -> str:
        return "primary"

    def get_ground_truth(self, sample: dict) -> str:
        gt = sample.get("answer", "")
        return gt.replace(self.EOS_TOKEN, "").strip()

    def extract_answer(self, prediction: str, sample: dict) -> str:
        return (prediction or "").strip()

    def evaluate_answer(self, prediction: str, sample: dict) -> dict:
        gt = self.get_ground_truth(sample)
        result = score_any(prediction or "", gt)
        # Binary correctness only when primary metric is exactly 1.0; the
        # partial-credit signal lives in result["primary"] and is averaged
        # via category_metric_sums (primary_metric = "primary").
        result["correct"] = bool(result["primary"] >= 1.0 - 1e-9)
        return result

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["id"] = row.get("id", "")
        sample["category"] = row.get("category", "")
        sample["length"] = int(row.get("length", 0))
        sample["n_channels"] = int(row.get("n_channels", 0))
        sample["question"] = row.get("question", "")
        sample["channel_names"] = list(row.get("channel_names", []))
        sample["channel_values"] = [
            np.asarray(v, dtype=np.float32) for v in row.get("channel_values", [])
        ]
        if self.variant == "precise":
            sample["is_precise"] = is_precise_gold(str(row["answer"]))
        else:
            sample["is_precise"] = False
        return sample
