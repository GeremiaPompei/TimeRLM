"""Project anomalyxl-precise rows into MCQ-style coarse rows that share the
SAME underlying time series.

Each emitted coarse row's ``id`` encodes its source precise row id and the MCQ
template it was projected through (e.g. ``paired__loc_L16384_0042__presence``),
so post-eval analysis can join paired coarse rows back to their precise
siblings and isolate the "stricter scoring" axis from the "different signals"
confounder that the independently-generated coarse shard has.

Projection rules (one precise row → one or two coarse rows):

- ``localize``                 → ``presence`` AND ``start_time``
                                 (start_time emits "No Anomaly" gold for absent rows)
- ``classify_with_evidence``   → ``categorization`` (gold = the kind label)
- ``measure_magnitude``        → ``magnitude`` (bin σ into the existing 4-bin scale)
- ``localize_all_channels``    → ``identification``
                                 (3 channels picked to cover the anomalous set; non-picked
                                  anomalous channels become documented distractors)
- ``lead_lag_with_magnitude``  → ``indicator``
                                 (direction maps to 3 of the 5 MCQ options;
                                  "Perfectly correlated" / "No anomaly..." never gold)

Stratified sampling across context length keeps L coverage balanced. Run:

    cd anomalyXL
    uv run python scripts/coarse_project.py \\
        --src datasets/anomalyxl-precise \\
        --out datasets/anomalyxl-coarse \\
        --per-category 40
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `anomalyxl` resolves
from anomalyxl.cli import _SCHEMA, _VALUE_DECIMALS  # noqa: E402
from anomalyxl.questions._time_buckets import (  # noqa: E402
    QUARTER_LABELS_START,
    bucket_label,
    index_to_bucket,
)
from anomalyxl.questions.categorization import (  # noqa: E402
    _KIND_TO_LABEL,
    _NO_ANOMALY_LABEL as _CAT_NO_ANOMALY,
    _QUESTION as _CAT_Q,
    _build_options as _categorization_options,
)
from anomalyxl.questions.identification import (  # noqa: E402
    _NONE as _ID_NONE,
    _fmt as _id_fmt,
    _build_options as _identification_options,
)
from anomalyxl.questions.indicator import (  # noqa: E402
    _OPTIONS as _IND_OPTIONS,
    _SEMANTIC as _IND_SEMANTIC,
    _QUESTION as _IND_Q,
)
from anomalyxl.questions.magnitude import (  # noqa: E402
    _BINS as _MAG_BINS,
    _QUESTION as _MAG_Q,
    _NO_ANOMALY as _MAG_NO_ANOMALY,
)
from anomalyxl.questions.presence import _QUESTION as _PRES_Q  # noqa: E402
from anomalyxl.questions.start_time import _QUESTION as _ST_Q  # noqa: E402


# Precise direction → indicator option index (lead/lag/independent only).
_LEAD_LAG_TO_INDICATOR = {
    "lead":        _IND_OPTIONS[0],   # "Leading indicator..."
    "lag":         _IND_OPTIONS[1],   # "Lagging indicator..."
    "independent": _IND_OPTIONS[3],   # "Not correlated..."
}


# ----- precise gold parser (same envelope as scorer) -----

def _parse_precise_gold(answer: str) -> tuple[str, dict]:
    cat, sep, payload = answer.partition(":")
    if not sep:
        raise ValueError(f"Not a precise gold envelope: {answer!r}")
    return cat, json.loads(payload)


# ----- record helpers -----

def _round_values(values_list):
    return [
        np.round(np.asarray(v, dtype=np.float32), _VALUE_DECIMALS).tolist()
        for v in values_list
    ]


def _base_record(src_row: dict, *, paired_template: str) -> dict:
    """Skeleton with the channel data + back-reference id. The caller fills
    in `tier`, `category`, `question`, `options`, `answer`, `semantic_class`,
    and any coupling/injection edits."""
    return {
        "id": f"paired__{src_row['id']}__{paired_template}",
        "tier": 0,  # filled by caller
        "category": paired_template,
        "length": int(src_row["length"]),
        "n_channels": int(src_row["n_channels"]),
        "n_channels_a": src_row.get("n_channels_a"),
        "n_channels_b": src_row.get("n_channels_b"),
        "question": "",
        "options": [],
        "answer": "",
        "semantic_class": "",
        "coupling_kind": src_row.get("coupling_kind"),
        "coupling_lag": src_row.get("coupling_lag"),
        "injection_kinds": list(src_row["injection_kinds"]),
        "injection_channels": list(src_row["injection_channels"]),
        "injection_starts": list(src_row["injection_starts"]),
        "injection_ends": list(src_row["injection_ends"]),
        "channel_names": list(src_row["channel_names"]),
        "channel_values": _round_values(src_row["channel_values"]),
        "seed": int(src_row["seed"]),
    }


# ----- per-category projection -----

def project_localize(src: dict, rng: np.random.Generator) -> list[dict]:
    _, payload = _parse_precise_gold(src["answer"])
    L = int(payload.get("_L", src["length"]))
    out: list[dict] = []

    # presence MCQ
    pres = _base_record(src, paired_template="presence")
    pres["tier"] = 1
    pres["question"] = _PRES_Q
    pres["answer"] = "Yes" if payload.get("present") else "No"
    pres["semantic_class"] = pres["answer"]
    opts = ["Yes", "No"]
    rng.shuffle(opts)
    pres["options"] = opts
    out.append(pres)

    # start_time MCQ
    st = _base_record(src, paired_template="start_time")
    st["tier"] = 2
    st["question"] = _ST_Q
    if not payload.get("present"):
        st["answer"] = "No Anomaly"
        st["semantic_class"] = "No Anomaly"
    else:
        bucket = index_to_bucket(int(payload["start"]), L)
        st["answer"] = bucket_label(QUARTER_LABELS_START[bucket], L, bucket)
        st["semantic_class"] = QUARTER_LABELS_START[bucket]
    options = [bucket_label(p, L, i) for i, p in enumerate(QUARTER_LABELS_START)]
    options.append("No Anomaly")
    rng.shuffle(options)
    st["options"] = options
    out.append(st)
    return out


def project_classify(src: dict, rng: np.random.Generator) -> list[dict]:
    _, payload = _parse_precise_gold(src["answer"])
    rec = _base_record(src, paired_template="categorization")
    rec["tier"] = 2
    rec["question"] = _CAT_Q
    gold_label = payload["kind"]  # already a label like "Level Shift"
    rec["answer"] = gold_label
    rec["semantic_class"] = gold_label  # raw label; downstream may collapse it
    rec["options"] = _categorization_options(gold_label, rng)
    return [rec]


def project_magnitude(src: dict, rng: np.random.Generator) -> list[dict]:
    _, payload = _parse_precise_gold(src["answer"])
    sigma = float(payload["magnitude_sigma"])
    # Bin σ. Bin ranges are closed-open; clamp the (rare) σ≥100 to the top bin
    # and σ<1 to the bottom bin so every row maps to one of the 4 labels.
    gold = _MAG_BINS[0][0]
    for name, lo, hi in _MAG_BINS:
        if lo <= sigma < hi:
            gold = name
            break
    else:
        if sigma >= _MAG_BINS[-1][2]:
            gold = _MAG_BINS[-1][0]
    rec = _base_record(src, paired_template="magnitude")
    rec["tier"] = 2
    rec["question"] = _MAG_Q
    rec["answer"] = gold
    rec["semantic_class"] = gold
    options = [name for name, _, _ in _MAG_BINS] + [_MAG_NO_ANOMALY]
    rng.shuffle(options)
    rec["options"] = options
    return [rec]


def project_localize_all_channels(src: dict, rng: np.random.Generator) -> list[dict]:
    _, payload = _parse_precise_gold(src["answer"])
    anomalies = payload.get("anomalies") or []
    anomalous_ch = sorted({a["channel"] for a in anomalies})
    K = len(anomalous_ch)
    if K > 3:
        # K is uniform over {0..3} by construction — this would be a generator bug.
        raise ValueError(f"localize_all_channels has K={K}>3 on row {src['id']}")

    all_ch = list(src["channel_names"])
    non_anomalous = [c for c in all_ch if c not in set(anomalous_ch)]
    rng.shuffle(non_anomalous)
    pick_count = 3 - K
    picked = sorted(anomalous_ch + non_anomalous[:pick_count])
    if len(picked) != 3:
        raise ValueError(f"Expected 3 picked channels, got {len(picked)} on {src['id']}")

    gold_subset = sorted(anomalous_ch)
    gold_label = _id_fmt(gold_subset)
    options = _identification_options(picked, gold_subset, rng)

    rec = _base_record(src, paired_template="identification")
    rec["tier"] = 2
    rec["question"] = (
        f"Among channels {picked[0]}, {picked[1]}, and {picked[2]}, "
        "which subset (if any) is anomalous? Other channels in the series may "
        "also be anomalous, but only these three are scored."
    )
    rec["answer"] = gold_label
    rec["semantic_class"] = str(K)
    rec["options"] = options
    return [rec]


def project_lead_lag(src: dict, rng: np.random.Generator) -> list[dict]:
    _, payload = _parse_precise_gold(src["answer"])
    direction = payload["direction"]
    if direction not in _LEAD_LAG_TO_INDICATOR:
        raise ValueError(f"Unmapped lead_lag direction {direction!r} on {src['id']}")
    gold_label = _LEAD_LAG_TO_INDICATOR[direction]
    options = list(_IND_OPTIONS)
    rng.shuffle(options)

    rec = _base_record(src, paired_template="indicator")
    rec["tier"] = 3
    rec["question"] = _IND_Q
    rec["answer"] = gold_label
    rec["semantic_class"] = _IND_SEMANTIC[gold_label]
    rec["options"] = options
    return [rec]


_PROJECTORS = {
    "localize":                project_localize,
    "classify_with_evidence":  project_classify,
    "measure_magnitude":       project_magnitude,
    "localize_all_channels":   project_localize_all_channels,
    "lead_lag_with_magnitude": project_lead_lag,
}


# ----- stratified sampler -----

def stratified_sample(rows: list[dict], category: str, n: int,
                      rng: np.random.Generator, *,
                      substrat_key=None) -> list[dict]:
    """Pick `n` rows of `category` evenly across context lengths.

    If ``substrat_key(row)`` is supplied, each length bucket is further split
    by the returned key and picks are spread evenly across (length × subkey)
    cells. Used to force direction balance on lead_lag, where 1/3 prior +
    small N would otherwise leave one direction ~2 SD underrepresented.
    """
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != category:
            continue
        by_len[int(r["length"])].append(r)
    if not by_len:
        return []
    lengths = sorted(by_len)
    per_len = n // len(lengths)
    remainder = n - per_len * len(lengths)
    picked: list[dict] = []
    for i, L in enumerate(lengths):
        bucket = by_len[L]
        take_L = min(per_len + (1 if i < remainder else 0), len(bucket))
        if substrat_key is None:
            bucket_sorted = sorted(bucket, key=lambda r: r["id"])
            idx = rng.permutation(len(bucket_sorted))
            picked.extend(bucket_sorted[j] for j in idx[:take_L])
            continue
        # Sub-stratify within this length bucket.
        sub: dict[str, list[dict]] = defaultdict(list)
        for r in bucket:
            sub[substrat_key(r)].append(r)
        keys = sorted(sub)
        per_key = take_L // len(keys)
        rem_key = take_L - per_key * len(keys)
        for ki, k in enumerate(keys):
            k_bucket = sorted(sub[k], key=lambda r: r["id"])
            k_take = min(per_key + (1 if ki < rem_key else 0), len(k_bucket))
            idx = rng.permutation(len(k_bucket))
            picked.extend(k_bucket[j] for j in idx[:k_take])
    return picked


def _lead_lag_direction_key(row: dict) -> str:
    payload = json.loads(row["answer"].partition(":")[2])
    return payload["direction"]


# ----- IO + main -----

def read_precise(parquet_path: Path) -> list[dict]:
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def write_coarse(rows: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    pq.write_table(table, out_dir / "data.parquet")
    return table.num_rows


def write_manifest(out_dir: Path, n_rows: int, src: Path, seed: int,
                   per_category: dict[str, int]) -> None:
    manifest = {
        "version": "0.3.0-paired",
        "artifact": "coarse-paired",
        "projected_from": str(src.resolve()),
        "seed": seed,
        "n_rows": n_rows,
        "per_precise_category_sampled": per_category,
        "projection_rules": {
            "localize":                ["presence", "start_time"],
            "classify_with_evidence":  ["categorization"],
            "measure_magnitude":       ["magnitude"],
            "localize_all_channels":   ["identification"],
            "lead_lag_with_magnitude": ["indicator"],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True,
                   help="Precise dataset dir containing data.parquet.")
    p.add_argument("--out", type=Path, required=True,
                   help="Destination dir for the coarse-paired shard.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-localize", type=int, default=40,
                   help="Number of precise `localize` rows to sample (each yields 2 coarse rows).")
    p.add_argument("--n-per-other", type=int, default=42,
                   help="Number of precise rows per non-localize category (1 coarse row each).")
    args = p.parse_args()

    src_parquet = args.src / "data.parquet"
    if not src_parquet.exists():
        raise SystemExit(f"No data.parquet at {args.src}")

    rng = np.random.default_rng(args.seed)
    all_rows = read_precise(src_parquet)

    sample_counts = {
        "localize":                args.n_localize,
        "classify_with_evidence":  args.n_per_other,
        "measure_magnitude":       args.n_per_other,
        "localize_all_channels":   args.n_per_other,
        "lead_lag_with_magnitude": args.n_per_other,
    }
    sampled: dict[str, list[dict]] = {}
    for cat, n in sample_counts.items():
        # lead_lag also stratifies by direction so the 3 outcomes are balanced
        # — at small N the 1/3 prior leaves noticeable variance otherwise.
        key = _lead_lag_direction_key if cat == "lead_lag_with_magnitude" else None
        sampled[cat] = stratified_sample(all_rows, cat, n, rng, substrat_key=key)
        print(f"  sampled {len(sampled[cat]):3d} / {n} from precise '{cat}'"
              + ("  (length × direction stratified)" if key else ""))

    coarse_rows: list[dict] = []
    for cat, rows in sampled.items():
        projector = _PROJECTORS[cat]
        for r in rows:
            coarse_rows.extend(projector(r, rng))

    n_written = write_coarse(coarse_rows, args.out)
    write_manifest(args.out, n_written, args.src, args.seed, sample_counts)

    print(f"\nWrote {n_written} coarse-paired rows to {args.out / 'data.parquet'}")
    print(f"Projection breakdown:")
    by_cat = defaultdict(int)
    for r in coarse_rows:
        by_cat[r["category"]] += 1
    for cat in sorted(by_cat):
        print(f"  {cat:20s} {by_cat[cat]:4d}")


if __name__ == "__main__":
    main()
