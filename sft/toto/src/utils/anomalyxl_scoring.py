"""AnomalyXL scoring — vendored from
``environments/timeseries_qa/ts_datasets/anomalyxl.py``.

Two answer formats coexist in AnomalyXL shards:

* **Precise** — gold is ``f"{category}:{json_payload}"`` with one of five
  categories (``localize``, ``classify_with_evidence``,
  ``measure_magnitude``, ``localize_all_channels``,
  ``lead_lag_with_magnitude``). Per-category scoring; primary metric
  varies (IoU, F1, composite).
* **Coarse** — gold is a bare option string (``"Yes"``, ``"No Anomaly"``,
  ``"Level Shift"``, etc.). Scoring is exact string match against any
  candidate extracted from the prediction.

`score_precise` and `score_coarse` each return ``{"primary": float ∈
[0,1], **extras}``. `score_any` dispatches by gold envelope shape so the
dataset class can stay format-agnostic.
"""

from __future__ import annotations

import math

import json
import re

PRECISE_SYSTEM_PROMPT = (
    "You analyze long-context time series and report grounded evidence. "
    "Your final answer MUST be a single JSON object on its own line, "
    "with no commentary, no markdown fences, no prose."
)

COARSE_SYSTEM_PROMPT = (
    "You are an expert at analyzing time series. You are given one or "
    "more channels of time-series data followed by a question. Respond with only "
    "the answer and nothing else."
)


# Sentinel `kind` marking the no-anomaly case in the unified detect+classify
# task (generalization sets). Must match `common.NONE_LABEL` in the generator.
_NONE_KIND = "No anomaly"

# Required prediction keys per precise category (used by the strict JSON parser).
# `localize` is special: `present` is always required; `start`/`end` are
# required ONLY when `present=true` — checked in `_score_localize`.
REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "localize": frozenset({"present"}),
    # `kind` is always required; `start`/`end` are needed only for a real kind
    # (the "No anomaly" case carries no window) — checked in the scorer.
    "classify_with_evidence": frozenset({"kind"}),
    "measure_magnitude": frozenset({"magnitude_sigma"}),
    "localize_all_channels": frozenset({"anomalies"}),
    "lead_lag_with_magnitude": frozenset({"direction", "lag_samples"}),
}

PRIMARY_METRIC: dict[str, str] = {
    "localize": "iou_or_correct_empty",
    "classify_with_evidence": "kind_x_iou",
    "measure_magnitude": "composite",
    "localize_all_channels": "f1",
    "lead_lag_with_magnitude": "composite",
}

# Every metric name a category may populate.
METRIC_NAMES: dict[str, tuple[str, ...]] = {
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


PRECISE_CATEGORIES = frozenset(REQUIRED_KEYS)


def parse_structured_pred(text: str, required: frozenset[str]) -> dict | None:
    """Extract the first balanced ``{...}`` object from ``text`` and parse it.

    Returns ``None`` on:
    - empty/no-brace input
    - JSON decode error
    - non-dict root
    - any required key missing
    """
    if not text:
        return None
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


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    if a_start >= a_end or b_start >= b_end:
        return 0.0
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _band_token(b: float) -> str:
    return f"{b:.1f}".replace(".", "p")


def _finite(value: object) -> bool:
    """True for a real, finite number. `json.loads` accepts null/NaN/Infinity/1e400,
    all of which would otherwise raise out of the scorer instead of scoring zero.
    Mirrors `_finite` in environments/timeseries_qa/ts_datasets/anomalyxl.py."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _zero_metrics(category: str) -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES[category]}


def _score_localize(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize")
    gold_present = bool(gold["present"])
    if pred is None:
        out["presence_correct"] = 0.0
        return out

    pred_present = bool(pred.get("present", False))
    out["presence_correct"] = 1.0 if pred_present == gold_present else 0.0

    if not gold_present and not pred_present:
        out["iou_or_correct_empty"] = 1.0
        out["iou"] = 1.0
        for band in (0.3, 0.5, 0.8):
            out[f"acc_at_iou_{_band_token(band)}"] = 1.0
        out["start_mae_fraction"] = 0.0
        out["end_mae_fraction"] = 0.0
        return out

    if gold_present and pred_present:
        if not (_finite(pred.get("start")) and _finite(pred.get("end"))):
            return out
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
    out["composite"] = max(0.0, 1.0 - rel_err / 0.5)
    return out


def _greedy_set_match(
    gold_events: list[dict], pred_events: list[dict], *, iou_threshold: float
) -> int:
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


def _score_localize_all_channels(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize_all_channels")
    gold_events = gold.get("anomalies", [])
    if pred is None:
        out["fn"] = float(len(gold_events))
        return out
    pred_events = pred.get("anomalies", []) if isinstance(pred.get("anomalies"), list) else []
    pred_events = [
        e for e in pred_events
        if isinstance(e, dict)
        and "channel" in e and "start" in e and "end" in e
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


_DISPATCH = {
    "localize": _score_localize,
    "classify_with_evidence": _score_classify_with_evidence,
    "measure_magnitude": _score_measure_magnitude,
    "localize_all_channels": _score_localize_all_channels,
    "lead_lag_with_magnitude": _score_lead_lag_with_magnitude,
}


def _split_precise_gold(gold: str) -> tuple[str, dict]:
    cat, sep, payload = gold.partition(":")
    if not sep:
        raise ValueError(f"Gold missing category prefix: {gold!r}")
    if cat not in _DISPATCH:
        raise KeyError(f"Unknown AnomalyXL category: {cat!r}")
    return cat, json.loads(payload)


def is_precise_gold(gold: str) -> bool:
    """Cheap shape test: precise golds start with ``<category>:{...}``."""
    if not isinstance(gold, str):
        return False
    cat, sep, payload = gold.partition(":")
    return bool(sep) and cat in _DISPATCH and payload.lstrip().startswith("{")


def score_precise(prediction: str, gold: str) -> dict:
    """Score a precise-format prediction.

    Returns a dict with ``primary`` ∈ [0,1] plus every category-specific
    extra metric (other-category extras are absent, not zeroed — caller
    aggregates).
    """
    cat, gold_payload = _split_precise_gold(gold)
    pred = parse_structured_pred(prediction or "", REQUIRED_KEYS[cat])
    metrics = _DISPATCH[cat](pred, gold_payload)
    primary = float(metrics.get(PRIMARY_METRIC[cat], 0.0))
    return {"primary": primary, "category": cat, **metrics}


def _coarse_candidates(text: str):
    """Yield candidate answer strings: the whole prediction, the
    ``"answer"`` field of any embedded JSON object, and an ``Answer:``
    line. Mirrors ``TSExamXL._answer_candidates``."""
    yield text
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        try:
            obj = json.loads(text[start:end])
        except (json.JSONDecodeError, TypeError, ValueError):
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            yield obj["answer"]
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
    if m:
        yield m.group(1).strip().strip("\"'")


def score_coarse(prediction: str, gold: str) -> dict:
    """Score a coarse-format prediction by exact string match against
    any extracted candidate."""
    gold = (gold or "").strip()
    for candidate in _coarse_candidates(prediction or ""):
        if candidate.strip() == gold:
            return {"primary": 1.0, "correct": 1.0}
    return {"primary": 0.0, "correct": 0.0}


def score_any(prediction: str, gold: str) -> dict:
    """Auto-dispatch by gold envelope shape."""
    if is_precise_gold(gold):
        return score_precise(prediction, gold)
    return score_coarse(prediction, gold)
