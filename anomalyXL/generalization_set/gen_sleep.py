"""Generalization set — Sleep-PSG (respiratory arousals). Splice into variable-length
single-anomaly windows in the AnomalyXL schema. Every window is a unified
`classify_with_evidence` row: positives carry the kind + window (scored kind x
IoU; localization is built in); the 20% negatives carry kind `No anomaly` and no
window.

Anomaly = one respiratory arousal bout (central_apnea / obstructive_apnea /
hypopnea / rera) with no other arousal event in the window. Gold [start,end) is
the bout's own native range (`bout_start_ms`/`bout_end_ms`) mapped to window
samples. Negatives are arousal-free *sleep* spans (non-Wake).
Emitted channels are configurable (`--channels`); the default keeps the few
channels that carry the respiratory-arousal signal. Records: test split only.

Source data is the PhysioNet 2018 Sleep-PSG challenge set. `--data-root` (or the
`SLEEP_DATA_ROOT` env var) must point at a directory holding
`ts_haystack/{arousals,arousal_free}/timelines/<rec>.parquet`,
`ts_haystack/participant_split.json`, and `training/<rec>/<rec>.{npy,hea}`:

  python anomalyXL/generalization_set/gen_sleep.py --data-root /path/to/sleep_psg --n 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import NONE_LABEL, classify_question, classify_row, make_id, write_dataset, zscore

DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", "data/sleep_psg"))
AROUSAL = DATA_ROOT / "ts_haystack/arousals/timelines"
AROUSAL_FREE = DATA_ROOT / "ts_haystack/arousal_free/timelines"
SPLIT = DATA_ROOT / "ts_haystack/participant_split.json"
SIG = DATA_ROOT / "training"


def _bind_data_root(root: Path) -> None:
    """Rebind the data paths (module-level so the readers below stay simple)."""
    global DATA_ROOT, AROUSAL, AROUSAL_FREE, SPLIT, SIG
    DATA_ROOT = root
    AROUSAL = root / "ts_haystack/arousals/timelines"
    AROUSAL_FREE = root / "ts_haystack/arousal_free/timelines"
    SPLIT = root / "ts_haystack/participant_split.json"
    SIG = root / "training"
    if not SPLIT.exists():
        raise SystemExit(f"--data-root {root} has no ts_haystack/participant_split.json")
HZ = 200
RESP = {"central_apnea", "obstructive_apnea", "hypopnea", "rera"}
# Channels carrying the strongest respiratory-arousal signal: nasal/oral airflow
# (apnea/hypopnea), thoracic + abdominal effort belts (central vs obstructive),
# and oxygen saturation (post-event desaturation). Override with --channels.
DEFAULT_CHANNELS = ["AIRFLOW", "CHEST", "ABD", "SaO2"]
KIND_LABEL = {"central_apnea": "Central Apnea", "obstructive_apnea": "Obstructive Apnea",
              "hypopnea": "Hypopnea", "rera": "RERA"}
KIND_LABELS = ["Central Apnea", "Obstructive Apnea", "Hypopnea", "RERA"]
DEFS = (
    "a `Central Apnea` is a cessation of airflow with absent respiratory effort; "
    "an `Obstructive Apnea` is a cessation of airflow despite continued respiratory "
    "effort against an obstructed airway; a `Hypopnea` is a marked reduction (not "
    "cessation) in airflow accompanied by oxygen desaturation or an arousal; a "
    "`RERA` (respiratory effort-related arousal) is a sequence of increasing "
    "respiratory effort that culminates in an arousal without meeting apnea or "
    "hypopnea criteria."
)


def ms2s(ms: float) -> int:
    return int(round(ms * HZ / 1000.0))


def channel_names(rec: str) -> list[str]:
    """All channel names for a record, in .npy column order (from the .hea)."""
    h = (SIG / rec / f"{rec}.hea").read_text().splitlines()
    n = int(h[0].split()[1])
    return [ln.split()[-1] for ln in h[1:1 + n]]


def extract(rec: str, ws: int, L: int, sel: list[str]):
    """z-normed window for the selected channels, or None if the record's montage
    is missing any of them (keeps the channel set identical across all samples)."""
    idx = {n: i for i, n in enumerate(channel_names(rec))}
    if any(c not in idx for c in sel):
        return None
    sig = load_sig(rec)[ws:ws + L]
    return [zscore(np.asarray(sig[:, idx[c]])) for c in sel]


def scan(rec: str):
    """Present-candidates (respiratory bouts) + clean sleep negatives, in samples."""
    t = pd.read_parquet(AROUSAL / f"{rec}.parquet").sort_values("bout_start_ms")
    cands = []
    if not t.empty:
        starts = t["bout_start_ms"].to_numpy()
        ends = t["bout_end_ms"].to_numpy()
        acts = t["bout_activity"].to_numpy()
        rec_end = int(t["recording_end_ms"].iloc[0])
        n = len(t)
        for j in range(n):
            if acts[j] not in RESP:
                continue
            c_lo = ms2s(ends[j - 1]) if j > 0 else 0
            c_hi = ms2s(starts[j + 1]) if j + 1 < n else ms2s(rec_end)
            cands.append((rec, ms2s(starts[j]), ms2s(ends[j]), str(acts[j]), c_lo, c_hi))
    # negatives: arousal-free spans that are real sleep (drop Wake)
    af = pd.read_parquet(AROUSAL_FREE / f"{rec}.parquet")
    af = af[af["bout_activity"] != "Wake"]
    gaps = [(rec, ms2s(s), ms2s(e)) for s, e in zip(af["bout_start_ms"], af["bout_end_ms"])]
    return cands, gaps


def load_sig(rec: str) -> np.ndarray:
    return np.load(SIG / rec / f"{rec}.npy", mmap_mode="r")


def lo_hi_sec(dur_s, corr_s, min_sec, max_sec, guard_s):
    """Window-length range (s): floor holds the event + guards, ceiling fits the
    isolating corridor (and the max-sec cap)."""
    return max(min_sec, dur_s + 2 * guard_s), min(max_sec, corr_s - 2 * guard_s)


def place_present(rng, cand, min_sec, max_sec, guard_s, sel, retries=6):
    """Return [classify_row] for one positive window (locate + classify), or None."""
    guard = int(round(guard_s * HZ))
    rec, g_lo, g_hi, act, c_lo, c_hi = cand
    dur_s, corr_s = (g_hi - g_lo) / HZ, (c_hi - c_lo) / HZ
    a, b = lo_hi_sec(dur_s, corr_s, min_sec, max_sec, guard_s)
    for _ in range(retries):
        L = int(round(rng.uniform(a, b) * HZ))
        lower = max(c_lo, g_hi - L + guard, g_lo + guard - L)
        upper = min(c_hi - L, g_lo - guard)
        if lower > upper:
            continue
        ws = int(rng.integers(lower, upper + 1))
        vals = extract(rec, ws, L, sel)
        if vals is None:
            return None
        return [classify_row(
            rid=make_id("sleep", rec, ws, L, "cls"), length=L, channel_names=sel,
            channel_values=vals, kind_label=KIND_LABEL[act], kind_snake=act,
            start=g_lo - ws, end=g_hi - ws, inj_channels=sel,
            question=classify_question(L, KIND_LABELS, DEFS), seed=int(rng.integers(1 << 31)))]
    return None


def place_absent(rng, gap, min_sec, max_sec, guard_s, sel, retries=6):
    """Return [classify_row] for one clean (no-anomaly) window, or None."""
    guard = int(round(guard_s * HZ))
    rec, lo, hi = gap
    gap_s = (hi - lo) / HZ
    for _ in range(retries):
        L = int(round(rng.uniform(min_sec, min(max_sec, gap_s - 2 * guard_s)) * HZ))
        if hi - guard - L < lo + guard:
            continue
        ws = int(rng.integers(lo + guard, hi - guard - L + 1))
        vals = extract(rec, ws, L, sel)
        if vals is None:
            return None
        return [classify_row(
            rid=make_id("sleep", rec, ws, L, "none"), length=L, channel_names=sel,
            channel_values=vals, kind_label=NONE_LABEL, kind_snake="none",
            question=classify_question(L, KIND_LABELS, DEFS), seed=int(rng.integers(1 << 31)))]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="Sleep-PSG root (default: $SLEEP_DATA_ROOT or ./data/sleep_psg)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "datasets/sleep-anomaly-short"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--present-frac", type=float, default=0.8)
    ap.add_argument("--min-sec", type=float, default=60.0)
    ap.add_argument("--max-sec", type=float, default=300.0)
    ap.add_argument("--guard-s", type=float, default=5.0)
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS),
                    help="comma-separated PSG channel names to emit (default: respiratory set)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    _bind_data_root(Path(args.data_root))
    rng = np.random.default_rng(args.seed)
    guard = int(round(args.guard_s * HZ))
    sel = [c.strip() for c in args.channels.split(",") if c.strip()]
    test = json.loads(SPLIT.read_text())["test"]

    cands, gaps = [], []
    for rec in test:
        if not (AROUSAL / f"{rec}.parquet").exists():
            continue
        c, g = scan(rec)
        cands += c
        gaps += g

    # usable present: window range [Llo, Lhi] non-empty (corridor fits + window holds event)
    usable = []
    for (rec, g_lo, g_hi, act, c_lo, c_hi) in cands:
        dur_s, corr_s = (g_hi - g_lo) / HZ, (c_hi - c_lo) / HZ
        a, b = lo_hi_sec(dur_s, corr_s, args.min_sec, args.max_sec, args.guard_s)
        if a <= b:
            usable.append((rec, g_lo, g_hi, act, c_lo, c_hi))
    gaps = [g for g in gaps if (g[2] - g[1]) / HZ >= args.min_sec + 2 * args.guard_s]

    n_present = int(round(args.n * args.present_frac))
    n_absent = args.n - n_present

    # balanced activity-class candidate stream (sorted keys -> deterministic
    # dict order so the per-class rng.shuffle sequence is reproducible)
    bya = {k: [c for c in usable if c[3] == k] for k in sorted(RESP)}
    for v in bya.values():
        rng.shuffle(v)
    order = sorted(RESP)
    ordered, i = [], 0
    while any(bya.values()):
        pool = bya[order[i % len(order)]]
        if pool:
            ordered.append(pool.pop())
        i += 1

    rows, n_win = [], 0
    for cand in ordered:
        if n_win >= n_present:
            break
        out = place_present(rng, cand, args.min_sec, args.max_sec, args.guard_s, sel)
        if out:
            rows.extend(out)
            n_win += 1
    rng.shuffle(gaps)
    n_neg = 0
    for gap in gaps:
        if n_neg >= n_absent:
            break
        out = place_absent(rng, gap, args.min_sec, args.max_sec, args.guard_s, sel)
        if out:
            rows.extend(out)
            n_neg += 1

    rng.shuffle(rows)
    write_dataset(rows, Path(args.out), {
        "dataset": "sleep-anomaly", "domain": "Sleep-PSG (real)", "source_hz": HZ,
        "anomaly": "single respiratory arousal bout; gold = native bout range",
        "channels": sel, "n_test_records": len(test), "seed": args.seed,
    })


if __name__ == "__main__":
    main()
