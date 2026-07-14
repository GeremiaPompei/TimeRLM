"""Shared schema + IO for the generalization-set generators.

Both generators splice a real recording into a single-anomaly window and emit a
row in the exact `anomalyxl-precise` `classify_with_evidence` schema, so the
existing `timeseries_qa` AnomalyXL env scores it unchanged (kind x IoU on
[start,end) sample indices, with a `No anomaly` kind carrying the negatives). The
only domain-specific work — finding an isolatable anomaly and reading the signal
— lives in the per-dataset generators.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Full anomalyxl-precise column set (fields we don't use are NaN/empty).
_COLUMNS = [
    "id", "tier", "category", "length", "n_channels", "n_channels_a",
    "n_channels_b", "question", "options", "answer", "semantic_class",
    "coupling_kind", "coupling_lag", "injection_kinds", "injection_channels",
    "injection_starts", "injection_ends", "channel_names", "channel_values",
    "seed",
]


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-channel z-normalization matching AnomalyXL `scene._normalize_unit`;
    a (near-)constant channel maps to zeros rather than blowing up."""
    x = x.astype(np.float32)
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - float(x.mean())) / std


def make_id(*parts) -> str:
    return hashlib.sha1("__".join(str(p) for p in parts).encode()).hexdigest()[:16]


# Sentinel kind for the no-anomaly case under the unified detect+classify prompt.
# Must match the sentinel used by the scorers in the verifiers env.
NONE_LABEL = "No anomaly"


def classify_question(length: int, kind_labels: list[str], defs_text: str) -> str:
    """Unified detect+classify template: the SAME prompt for anomalous and clean
    windows, with the domain's kinds enumerated plus a ``No anomaly`` option so
    detection is measured under one prompt (no separate present/absent task)."""
    kinds = ", ".join(f'"{k}"' for k in kind_labels)
    return (
        "Determine whether the time series contains an anomaly and, if so, "
        "identify it and its window. Respond with exactly one JSON object on a "
        "single line, no other text:\n"
        '{"kind": <label>, "start": <int>, "end": <int>}\n'
        f'`kind` must be one of: {kinds}, or "{NONE_LABEL}" if the series '
        'contains no anomaly. `start` is inclusive, `end` is exclusive; if '
        f'"{NONE_LABEL}", omit them. Each anomaly kind is defined as follows: '
        f"{defs_text} (Here L = {length}.)"
    )


def classify_row(
    *,
    rid: str,
    length: int,
    channel_names: list[str],
    channel_values: list[np.ndarray],
    kind_label: str,
    kind_snake: str,
    question: str,
    seed: int,
    start: int | None = None,
    end: int | None = None,
    inj_channels: list[str] | None = None,
) -> dict:
    """One `classify_with_evidence` row, scored ``kind_x_iou`` = kind-match × IoU.
    For the no-anomaly case pass ``kind_label=NONE_LABEL`` and omit start/end —
    gold then has no window and the scorer credits a matching ``No anomaly`` pred."""
    none = start is None or end is None
    if none:
        payload = {"kind": kind_label, "_L": int(length)}
        inj_kinds, inj_ch, inj_starts, inj_ends = [], [], [], []
    else:
        payload = {"kind": kind_label, "start": int(start), "end": int(end), "_L": int(length)}
        inj_kinds, inj_ch = [kind_snake], list(inj_channels or [])
        inj_starts, inj_ends = [int(start)], [int(end)]
    return {
        "id": rid,
        "tier": 2,
        "category": "classify_with_evidence",
        "length": int(length),
        "n_channels": len(channel_names),
        "n_channels_a": np.nan,
        "n_channels_b": np.nan,
        "question": question,
        "options": [],
        "answer": "classify_with_evidence:" + json.dumps(payload, sort_keys=True),
        "semantic_class": f"classify_{kind_snake}_L{length}",
        "coupling_kind": np.nan,
        "coupling_lag": np.nan,
        "injection_kinds": inj_kinds,
        "injection_channels": inj_ch,
        "injection_starts": inj_starts,
        "injection_ends": inj_ends,
        "channel_names": list(channel_names),
        "channel_values": [np.asarray(v, dtype=np.float32) for v in channel_values],
        "seed": int(seed),
    }


def write_dataset(rows: list[dict], out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_parquet(out_dir / "data.parquet")
    n_none = int(df["answer"].str.contains(f'"kind": "{NONE_LABEL}"').sum())
    n_anom = len(df) - n_none
    meta = {**meta, "n_rows": len(rows), "n_anomaly": n_anom, "n_no_anomaly": n_none,
            "lengths_samples": sorted({r["length"] for r in rows})}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir/'data.parquet'}  ({len(rows)} rows: "
          f"{n_anom} anomaly / {n_none} no-anomaly)  "
          f"length samples: {df['length'].min()}-{df['length'].max()}")
