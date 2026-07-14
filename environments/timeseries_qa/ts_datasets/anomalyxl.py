"""AnomalyXL — evidence-grounded long-context anomaly QA.

Loads parquet shards produced by
``anomalyXL/scripts/generate_anomalyxl_sweep.py`` (main shard).
Every gold answer is encoded as ``f"{category}:{json}"``; the env-side
parser extracts a single JSON object from the model output, requires
the per-category schema, and scores per a category-specific metric
dispatch.

Key design choices, all locked per ``AnomalyXL_PLAN.md`` + the
implementation plan:

- **Strict JSON, no echoed-prose fallback.** Format failure is a zero —
  it's the same "grounding requirement" being measured.
- **No L-stuffing of the gold prefix.** ``_L`` lives inside the JSON
  payload so MAE-as-fraction-of-L is computable from the gold alone.
- **Per-category metric union.** ``extra_metric_names`` is the union of
  metrics declared by categories actually present in the loaded shard;
  irrelevant metrics return 0 per row, the rubric averages them as the
  env always does.
- **Set-matching** for ``localize_all_channels`` is greedy by IoU on
  channel-matched pairs (plan §3.4).
- **No-anomaly is first-class for ``localize``.** Both-absent → 1.0;
  mismatch → 0.0; both-present → standard IoU.

This file deliberately reuses ``_resolve_path`` and ``_VALUE_DECIMALS``
from ``anomalyxl.py`` so the loader scaffold stays in one place.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset

from .base import Sample, TimeSeriesQADataset, strip_reasoning
from .anomalyxl_coarse import AnomalyXLCoarse, _resolve_path

_VALUE_DECIMALS = AnomalyXLCoarse._VALUE_DECIMALS

__all__ = ["AnomalyXL"]


# Default to the Localize (precise) test split of the released AnomalyXL monorepo, so
# a fresh clone works with no data prep. A locally generated directory always wins when
# passed explicitly as `repo_or_path` (or via $ANOMALYXL_DATA_ROOT); other splits are
# reachable as `nz00shuuuu/AnomalyXL#Localize/{train,val,test}` — see `_resolve_path`.
_DEFAULT_REPO_OR_PATH = "nz00shuuuu/AnomalyXL#Localize/test"

# Sentinel `kind` for the no-anomaly case in the unified detect+classify task
# (generalization sets). Must match `common.NONE_LABEL` in the generator.
_NONE_KIND = "No anomaly"

# Required prediction keys per category (used by the strict JSON parser).
# `localize` is special: `present` is always required; `start`/`end` are
# required ONLY when `present=true` — checked in `_score_localize`.
_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "localize": frozenset({"present"}),
    # `kind` always required; `start`/`end` only for a real kind (the
    # "No anomaly" case carries no window) — checked in the scorer.
    "classify_with_evidence": frozenset({"kind"}),
    "measure_magnitude": frozenset({"magnitude_sigma"}),
    "localize_all_channels": frozenset({"anomalies"}),
    "lead_lag_with_magnitude": frozenset({"direction", "lag_samples"}),
}

_PRIMARY_METRIC: dict[str, str] = {
    "localize": "iou_or_correct_empty",
    "classify_with_evidence": "kind_x_iou",
    "measure_magnitude": "composite",
    "localize_all_channels": "f1",
    "lead_lag_with_magnitude": "composite",
}

# Every metric name a category may populate. Used to build
# `extra_metric_names` as a union over the loaded shard's categories.
_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "localize": (
        "iou_or_correct_empty",
        "presence_correct",
        "iou",
        "acc_at_iou_0p3",
        "acc_at_iou_0p5",
        "acc_at_iou_0p8",
        "start_mae_fraction",
        "end_mae_fraction",
    ),
    "classify_with_evidence": (
        "kind_x_iou",
        "kind_correct",
        "iou",
        "kind_and_iou_above_0p5",
    ),
    "measure_magnitude": (
        "composite",
        "acc_at_rel_25pct",
        "acc_at_rel_10pct",
        "acc_at_rel_50pct",
        "rel_err",
    ),
    "localize_all_channels": (
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    ),
    "lead_lag_with_magnitude": (
        "composite",
        "direction_correct",
        "lag_acc_at_1pct",
        "lag_acc_at_5pct",
        "lag_mae_fraction",
    ),
}


def parse_structured_pred(text: str, required: frozenset[str]) -> dict | None:
    """Extract the first balanced ``{...}`` object from ``text`` and parse it.

    Returns ``None`` on:
    - empty/no-brace input
    - JSON decode error
    - non-dict root (e.g. the model returns a list)
    - any required key missing

    NO echoed-prose fallback — format failure is a zero by design.
    """
    if not text:
        return None
    # Drop <think>...</think> so the first balanced object found is the model's
    # actual answer, not a `{...}` it wrote while reasoning. No-op without a think
    # block (verified to leave current precise scores unchanged).
    text = strip_reasoning(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if not required.issubset(obj):
        return None
    return obj


# --------------------------------------------------------------------- scoring


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU on half-open intervals ``[start, end)``. Returns 0 if either is
    degenerate (``start >= end``) or both intervals are disjoint."""
    if a_start >= a_end or b_start >= b_end:
        return 0.0
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _zero_metrics(category: str) -> dict[str, float]:
    """Floor metrics for a category — every metric the category declares set to 0."""
    return {name: 0.0 for name in _METRIC_NAMES[category]}


def _finite(value: object) -> bool:
    """True for a real, finite number.

    Guards every numeric field a model may hand us. ``json.loads`` happily accepts
    ``null``, ``NaN``, ``Infinity`` and ``1e400``, and a model that decides there is
    no window will readily emit ``{"present": true, "start": null, "end": null}`` —
    all of which used to raise (TypeError/ValueError/OverflowError) out of the
    scorer instead of scoring zero. A non-numeric value is a format failure, and a
    format failure is a zero, not a crash. ``bool`` is excluded on purpose: it is an
    ``int`` subclass, and ``True`` is not a sample index.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _score_localize(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize")
    gold_present = bool(gold["present"])
    if pred is None:
        # Empty/absent on the gold side rewards format failure with 0 too —
        # we want the model to commit explicitly.
        out["presence_correct"] = 0.0
        return out

    pred_present = bool(pred.get("present", False))
    out["presence_correct"] = 1.0 if pred_present == gold_present else 0.0

    if not gold_present and not pred_present:
        # Both absent — correct empty.
        out["iou_or_correct_empty"] = 1.0
        out["iou"] = 1.0  # convention: empty-empty is a "perfect match"
        for band in (0.3, 0.5, 0.8):
            out[f"acc_at_iou_{_band_token(band)}"] = 1.0
        out["start_mae_fraction"] = 0.0
        out["end_mae_fraction"] = 0.0
        return out

    if gold_present and pred_present:
        if not (_finite(pred.get("start")) and _finite(pred.get("end"))):
            return out  # presence_correct=1 but no usable window → can't score IoU
        L = int(gold.get("_L", 0))
        iou = _iou(gold["start"], gold["end"], int(pred["start"]), int(pred["end"]))
        out["iou"] = iou
        out["iou_or_correct_empty"] = iou
        for band in (0.3, 0.5, 0.8):
            out[f"acc_at_iou_{_band_token(band)}"] = 1.0 if iou >= band else 0.0
        if L > 0:
            out["start_mae_fraction"] = abs(int(pred["start"]) - gold["start"]) / L
            out["end_mae_fraction"] = abs(int(pred["end"]) - gold["end"]) / L
        else:
            out["start_mae_fraction"] = 1.0
            out["end_mae_fraction"] = 1.0
        return out

    # Mismatch (one says present, the other absent) — zero on everything window-related.
    return out


def _score_classify_with_evidence(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("classify_with_evidence")
    gold_kind = str(gold["kind"])
    gold_none = gold_kind == _NONE_KIND
    if pred is None:
        return out
    pred_kind = str(pred.get("kind", ""))
    pred_none = pred_kind == _NONE_KIND
    kind_correct = 1.0 if pred_kind == gold_kind else 0.0
    out["kind_correct"] = kind_correct
    # "No anomaly" is the empty case (no window): full credit only when both
    # agree; a kind/none mismatch is a false negative/positive (iou stays 0).
    if gold_none or pred_none:
        if gold_none and pred_none:
            out["iou"] = 1.0
            out["kind_x_iou"] = 1.0
            out["kind_and_iou_above_0p5"] = 1.0
        return out
    # Reached only when BOTH gold and pred name a real kind (the "No anomaly" case
    # returned above with full credit), so a window is required here.
    if _finite(pred.get("start")) and _finite(pred.get("end")):
        iou = _iou(gold["start"], gold["end"], int(pred["start"]), int(pred["end"]))
    else:
        iou = 0.0
    out["iou"] = iou
    out["kind_x_iou"] = kind_correct * iou
    out["kind_and_iou_above_0p5"] = 1.0 if (kind_correct == 1.0 and iou >= 0.5) else 0.0
    return out


def _score_measure_magnitude(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("measure_magnitude")
    if pred is None:
        out["rel_err"] = 1.0
        return out
    raw = pred.get("magnitude_sigma")
    if not _finite(raw):
        out["rel_err"] = 1.0
        return out
    pred_sigma = float(raw)
    gold_sigma = float(gold["magnitude_sigma"])
    if gold_sigma <= 0:
        out["rel_err"] = 1.0
        return out
    rel_err = abs(pred_sigma - gold_sigma) / gold_sigma
    out["rel_err"] = rel_err
    out["acc_at_rel_10pct"] = 1.0 if rel_err <= 0.10 else 0.0
    out["acc_at_rel_25pct"] = 1.0 if rel_err <= 0.25 else 0.0
    out["acc_at_rel_50pct"] = 1.0 if rel_err <= 0.50 else 0.0
    # Smooth primary metric: linear ramp from 1.0 (exact) to 0.0 (50% relative error).
    # Replaces the hard 25% cutoff so a near-miss (e.g. 26% off) still earns most credit
    # instead of dropping to zero on a single percentage-point threshold.
    out["composite"] = max(0.0, 1.0 - rel_err / 0.5)
    return out


def _score_localize_all_channels(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize_all_channels")
    gold_events = gold.get("anomalies", [])
    if pred is None:
        # Pred missing — every gold event is a false negative; no false positives.
        out["fn"] = float(len(gold_events))
        # Empty gold + missing pred: precision/recall undefined → treat as 0.
        return out
    pred_events = pred.get("anomalies", []) if isinstance(pred.get("anomalies"), list) else []
    # Strict event shape — drop malformed predictions silently (they don't count as FPs;
    # they just don't match anything).
    pred_events = [
        e for e in pred_events
        if isinstance(e, dict)
        and "channel" in e
        and _finite(e.get("start")) and _finite(e.get("end"))
    ]

    tp = _greedy_set_match(gold_events, pred_events, iou_threshold=0.3)
    n_gold = len(gold_events)
    n_pred = len(pred_events)
    fp = max(0, n_pred - tp)
    fn = max(0, n_gold - tp)
    out["tp"] = float(tp)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    if n_gold == 0 and n_pred == 0:
        out["precision"] = 1.0
        out["recall"] = 1.0
        out["f1"] = 1.0
        return out
    if n_pred == 0 or n_gold == 0:
        return out
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold if n_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1
    return out


def _greedy_set_match(gold_events: list[dict], pred_events: list[dict],
                      *, iou_threshold: float) -> int:
    """Greedy IoU-descending matching across channel-paired events. Returns TP count.

    Each predicted event matches at most one gold event with the same
    ``channel`` and IoU > ``iou_threshold``. We sort candidate (gold,
    pred) pairs by descending IoU and greedily assign in that order.
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold_events):
        g_start, g_end = int(g["start"]), int(g["end"])
        g_ch = g["channel"]
        for pi, p in enumerate(pred_events):
            if str(p["channel"]) != str(g_ch):
                continue
            iou = _iou(g_start, g_end, int(p["start"]), int(p["end"]))
            if iou > iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    tp = 0
    for _, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        tp += 1
    return tp


def _score_lead_lag_with_magnitude(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("lead_lag_with_magnitude")
    if pred is None:
        out["lag_mae_fraction"] = 1.0
        return out
    pred_dir = str(pred.get("direction", ""))
    gold_dir = str(gold["direction"])
    direction_correct = 1.0 if pred_dir == gold_dir else 0.0
    out["direction_correct"] = direction_correct
    L = int(gold.get("_L", 0))

    if gold_dir == "independent":
        if direction_correct == 1.0:
            out["lag_mae_fraction"] = 0.0
            out["composite"] = 1.0
            out["lag_acc_at_1pct"] = 1.0
            out["lag_acc_at_5pct"] = 1.0
        else:
            out["lag_mae_fraction"] = 1.0
        return out

    # Gold direction is "lead" or "lag".
    pred_lag = pred.get("lag_samples")
    if not _finite(pred_lag) or L <= 0:
        out["lag_mae_fraction"] = 1.0
        return out
    if direction_correct != 1.0:
        out["lag_mae_fraction"] = 1.0
        return out
    lag_mae = abs(int(pred_lag) - int(gold["lag_samples"])) / L
    out["lag_mae_fraction"] = lag_mae
    out["lag_acc_at_1pct"] = 1.0 if lag_mae <= 0.01 else 0.0
    out["lag_acc_at_5pct"] = 1.0 if lag_mae <= 0.05 else 0.0
    out["composite"] = max(0.0, 1.0 - min(1.0, lag_mae / 0.05))
    return out


def _band_token(b: float) -> str:
    return f"{b:.1f}".replace(".", "p")


_DISPATCH = {
    "localize": _score_localize,
    "classify_with_evidence": _score_classify_with_evidence,
    "measure_magnitude": _score_measure_magnitude,
    "localize_all_channels": _score_localize_all_channels,
    "lead_lag_with_magnitude": _score_lead_lag_with_magnitude,
}


def _split_gold(gold: str) -> tuple[str, dict]:
    """``"category:<json>"`` → ``(category, payload_dict)``. Raises on malformed."""
    cat, sep, payload = gold.partition(":")
    if not sep:
        raise ValueError(f"Gold missing category prefix: {gold!r}")
    if cat not in _DISPATCH:
        raise KeyError(f"Unknown AnomalyXL category: {cat!r}")
    return cat, json.loads(payload)


# --------------------------------------------------------------------- dataset class


class AnomalyXL(TimeSeriesQADataset):
    name = "anomalyxl"
    splits = ("train", "eval")
    default_split = "train"
    default_eval_split = "eval"

    # Where ``data.parquet`` lives by default, and the env var that overrides it.
    # Subclasses (e.g. the real-data generalization sets) point these elsewhere
    # while inheriting the whole schema + scoring stack unchanged.
    default_repo_or_path = _DEFAULT_REPO_OR_PATH
    env_var = "ANOMALYXL_DATA_ROOT"

    system_prompt = (
        "You analyze long-context time series and report grounded evidence. "
        "Your final answer MUST be a single JSON object on its own line, "
        "with no commentary, no markdown fences, no prose."
    )
    rlm_values_note = (
        "Paired questions prefix channels with `A_` / `B_`. "
        "Multi-channel questions present 16 channels named `ch_00..ch_15`. "
    )
    rlm_answer_format = (
        "set answer['content'] to a single JSON object matching the question's "
        "schema exactly (e.g. {\"start\": 12, \"end\": 87}), then set "
        "answer['ready'] = True. Format failures score zero."
    )
    rlm_sub_llm_nudge = (
        "Series can be up to ~131k samples; for questions that benefit from "
        "narrative description of multiple sections (e.g. localizing a brief "
        "anomaly in a long series), prefer `llm_batch` over a single mega-prompt: "
        "split the series into segments, save each segment's plot or summary "
        "stats to a file, then send a batch of section-level questions to "
        "sub-LLMs in parallel. Use it when you need semantic judgment per "
        "section, not when numpy stats suffice. "
    )

    def __init__(
        self,
        repo_or_path: str | None = None,
        eval_size: int = 200,
        seed: int = 42,
        length_min: int | None = None,
        length_max: int | None = None,
        n_channels_max: int | None = None,
        categories: tuple[str, ...] | list[str] | None = None,
        ids: tuple[str, ...] | list[str] | None = None,
        mode: str = "text",
        text_max_samples_per_channel: int = 8192,
        text_total_samples: int | None = None,
        image_max_samples_per_channel: int = 16384,
    ):
        """``repo_or_path``: local directory or HF dataset repo id holding
        the AnomalyXL ``data.parquet``. Falls back to ``$ANOMALYXL_DATA_ROOT``
        and then to ``_DEFAULT_REPO_OR_PATH``.

        Filtering knobs (``length_min/max``, ``n_channels_max``, ``categories``)
        let users sub-select slices without regenerating the parquet — useful
        for ablations like "all categories at L=131k only" or "drop the
        multi-channel case for a single-channel-only ablation".
        """
        self.repo_or_path = (
            repo_or_path
            or (os.environ.get(self.env_var) if self.env_var else None)
            or self.default_repo_or_path
        )
        self.eval_size = eval_size
        self.seed = seed
        self.length_min = length_min
        self.length_max = length_max
        self.n_channels_max = n_channels_max
        self.categories = tuple(categories) if categories else None
        # `ids` lets callers pick a specific row subset (e.g., the errored rows
        # from a prior run) without writing a temp parquet. Applied alongside
        # other filters, before the seed shuffle.
        self.ids = tuple(ids) if ids else None
        self.mode = mode
        self.text_max_samples_per_channel = text_max_samples_per_channel
        # Optional total-samples budget across all channels (avoids
        # multi-channel prompts blowing past Qwen-class 32k input windows).
        # When set, per-channel cap = max(256, text_total_samples // n_channels).
        self.text_total_samples = text_total_samples
        self.image_max_samples_per_channel = image_max_samples_per_channel
        self._frame: pd.DataFrame | None = None
        self._train_df: pd.DataFrame | None = None
        self._eval_df: pd.DataFrame | None = None
        self._extra_metric_names: tuple[str, ...] = ()

    @property
    def extra_metric_names(self) -> tuple[str, ...]:  # type: ignore[override]
        # Computed lazily so the parquet is touched exactly once, on first
        # split build. Verifiers calls this once per env init.
        if not self._extra_metric_names:
            self._compute_extra_metric_names()
        return self._extra_metric_names

    def _compute_extra_metric_names(self) -> None:
        df = self._load_frame()
        present = sorted(df["category"].unique())
        names: list[str] = []
        for cat in present:
            if cat not in _METRIC_NAMES:
                # Defensive: the shard included a non-AnomalyXL category (e.g.
                # the user pointed at a anomalyxl-fab shard). Skip rather than
                # raise — the env will still score via score_answer.
                continue
            for name in _METRIC_NAMES[cat]:
                if name not in names:
                    names.append(name)
        self._extra_metric_names = tuple(names)

    def _load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        root = _resolve_path(self.repo_or_path)
        parquet = root / "data.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"No data.parquet under {root}")
        df = pd.read_parquet(parquet)
        if self.length_min is not None:
            df = df[df["length"] >= int(self.length_min)]
        if self.length_max is not None:
            df = df[df["length"] <= int(self.length_max)]
        if self.n_channels_max is not None:
            df = df[df["n_channels"] <= int(self.n_channels_max)]
        if self.categories is not None:
            df = df[df["category"].isin(self.categories)]
        if self.ids is not None:
            df = df[df["id"].isin(self.ids)]
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        self._frame = df
        return df

    def _split_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._train_df is not None and self._eval_df is not None:
            return self._train_df, self._eval_df
        df = self._load_frame()
        n_eval = min(self.eval_size, len(df))
        self._eval_df = df.iloc[:n_eval].reset_index(drop=True)
        self._train_df = df.iloc[n_eval:].reset_index(drop=True)
        return self._train_df, self._eval_df

    def load_split(self, split: str) -> Dataset:
        train, ev = self._split_dfs()
        return Dataset.from_pandas(ev if split == "eval" else train, preserve_index=False)

    def to_sample(self, row: dict) -> Sample:
        names = list(row["channel_names"])
        values = [list(v) for v in row["channel_values"]]
        series = {n: [round(float(x), _VALUE_DECIMALS) for x in v] for n, v in zip(names, values)}
        original_length = int(row["length"])
        cap = self._decimation_cap(n_channels=len(names))
        if cap is not None and original_length > cap:
            step = max(1, original_length // cap)
            series = {n: v[::step] for n, v in series.items()}
        # AnomalyXL has no options block — questions are open JSON.
        # `length` is the *original* sample count, kept on the sample so the
        # image renderer can plot the (possibly decimated) values on the true
        # index axis. Without this the model reads the plot at decimated
        # positions and the gold's true-index answers look wrong.
        return {
            "series": series,
            "question": row["question"],
            "answer": str(row["answer"]),
            "length": original_length,
        }

    def _decimation_cap(self, n_channels: int = 1) -> int | None:
        if self.mode == "text":
            if self.text_total_samples is not None and n_channels > 0:
                per_channel = max(256, self.text_total_samples // n_channels)
                return min(self.text_max_samples_per_channel, per_channel)
            return self.text_max_samples_per_channel
        if self.mode == "image":
            return self.image_max_samples_per_channel
        return None  # RLM mode: full series via sandbox file

    # --------------- scoring API

    def score_answer(self, parsed_text: str, gold: str) -> float:
        try:
            cat, gold_payload = _split_gold(gold)
        except (KeyError, ValueError, json.JSONDecodeError):
            return 0.0
        pred = parse_structured_pred(parsed_text or "", _REQUIRED_KEYS[cat])
        metrics = _DISPATCH[cat](pred, gold_payload)
        primary = _PRIMARY_METRIC[cat]
        return float(metrics.get(primary, 0.0))

    def score_metrics(self, parsed_text: str, gold: str) -> dict[str, float]:
        try:
            cat, gold_payload = _split_gold(gold)
        except (KeyError, ValueError, json.JSONDecodeError):
            return {name: 0.0 for name in self.extra_metric_names}
        pred = parse_structured_pred(parsed_text or "", _REQUIRED_KEYS[cat])
        per_cat = _DISPATCH[cat](pred, gold_payload)
        # Project into the union so the rubric column set is stable across rows.
        out = {name: 0.0 for name in self.extra_metric_names}
        for k, v in per_cat.items():
            if k in out:
                out[k] = float(v)
        return out
