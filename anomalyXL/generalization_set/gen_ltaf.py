"""Generalization set — LTAF (ECG). Splice the LTAF haystack into variable-length
single-anomaly windows in the AnomalyXL schema. Every window is a unified
`classify_with_evidence` row: positives carry the kind + window (scored kind x
IoU; localization is built in); the 20% negatives carry kind `No anomaly` and no
window.

Anomaly = one premature beat (V=PVC or A=APC, balanced) inside an NSR bout, with
no other non-`N` beat in the window. The gold [start,end) is the RR bracket
(previous beat -> next beat) around the anomalous beat — the only span the beat
annotations support, capturing the premature beat + compensatory pause. Negatives
are clean NSR windows with no non-`N` beat. Records: test split only.

Source data is the PhysioNet Long Term AF Database (ltafdb). `--data-root` (or the
`LTAF_DATA_ROOT` env var) must point at a directory holding
`ltaf_haystack/{beat_timelines,timelines}/<rec>.parquet` and `training/<rec>/<rec>.npy`:

  python anomalyXL/generalization_set/gen_ltaf.py --data-root /path/to/ltafdb --n 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import NONE_LABEL, classify_question, classify_row, make_id, write_dataset, zscore

DATA_ROOT = Path(os.environ.get("LTAF_DATA_ROOT", "data/ltafdb"))
BEAT = DATA_ROOT / "ltaf_haystack/beat_timelines"
RHY = DATA_ROOT / "ltaf_haystack/timelines"
SIG = DATA_ROOT / "training"


def _bind_data_root(root: Path) -> None:
    """Rebind the data paths (module-level so the readers below stay simple)."""
    global DATA_ROOT, BEAT, RHY, SIG
    DATA_ROOT = root
    BEAT = root / "ltaf_haystack/beat_timelines"
    RHY = root / "ltaf_haystack/timelines"
    SIG = root / "training"
    if not BEAT.is_dir():
        raise SystemExit(f"--data-root {root} has no ltaf_haystack/beat_timelines/")
HZ = 128
TEST = ["100", "104", "105", "11", "200", "32", "48", "49", "68"]
ANOM_SYMS = {"V", "A"}
CHANNELS = ["ECG_0", "ECG_1"]
KIND_SNAKE = {"V": "pvc", "A": "apc"}
KIND_LABEL = {"V": "Premature Ventricular Contraction (PVC)",
              "A": "Premature Atrial Contraction (APC)"}
KIND_LABELS = ["Premature Ventricular Contraction (PVC)", "Premature Atrial Contraction (APC)"]
DEFS = (
    "a `Premature Ventricular Contraction (PVC)` is an early, wide and bizarre "
    "QRS complex arising from the ventricles, not preceded by a P wave and "
    "typically followed by a compensatory pause; a `Premature Atrial Contraction "
    "(APC)` is an early beat arising from the atria, preceded by an abnormal P "
    "wave with a usually narrow QRS, followed by a partial compensatory pause."
)


def scan(rec: str):
    """Yield present-candidates and clean-gap negatives for one record."""
    beats = pd.read_parquet(BEAT / f"{rec}.parquet", columns=["sample", "symbol"]).sort_values("sample")
    rhy = pd.read_parquet(RHY / f"{rec}.parquet")
    nsr = rhy[rhy["activity"] == "NSR"][["start_sample", "end_sample"]].to_numpy()
    samp = beats["sample"].to_numpy()
    sym = beats["symbol"].to_numpy()
    nonN = samp[sym != "N"]

    cands, gaps = [], []
    for bs, be in nsr:
        bi = np.flatnonzero((samp >= bs) & (samp < be))  # beats in this bout
        if len(bi) < 3:
            continue
        in_bout_anom = nonN[(nonN >= bs) & (nonN < be)]
        for k, gi in enumerate(bi):
            if sym[gi] not in ANOM_SYMS:
                continue
            a = int(samp[gi])
            # corridor: clipped by nearest *any* non-N beat and bout edges
            lo_a = in_bout_anom[in_bout_anom < a]
            hi_a = in_bout_anom[in_bout_anom > a]
            c_lo = int(lo_a.max()) if len(lo_a) else int(bs)
            c_hi = int(hi_a.min()) if len(hi_a) else int(be)
            # RR bracket gold: previous -> next beat (any symbol)
            g_lo = int(samp[bi[k - 1]]) if k > 0 else a
            g_hi = int(samp[bi[k + 1]]) if k + 1 < len(bi) else a + 1
            cands.append((rec, a, sym[gi], c_lo, c_hi, g_lo, g_hi))
        # clean gaps for negatives: spans between consecutive non-N beats (+ edges)
        edges = np.concatenate(([bs], in_bout_anom, [be]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            gaps.append((rec, int(lo), int(hi)))
    return cands, gaps


def load_sig(rec: str) -> np.ndarray:
    return np.load(SIG / rec / f"{rec}.npy", mmap_mode="r")


def place_present(rng, cand, min_sec, max_sec, guard_s, retries=6):
    """Return [classify_row] for one positive window (locate + classify), or None."""
    guard = int(round(guard_s * HZ))
    rec, a, s, c_lo, c_hi, g_lo, g_hi = cand
    corr_s = (c_hi - c_lo) / HZ
    for _ in range(retries):
        L = int(round(rng.uniform(min_sec, min(max_sec, corr_s - 2 * guard_s)) * HZ))
        lower = max(c_lo, g_hi - L, a + guard - L)
        upper = min(c_hi - L, g_lo, a - guard)
        if lower > upper:
            continue
        ws = int(rng.integers(lower, upper + 1))
        sig = load_sig(rec)[ws:ws + L, :2]
        vals = [zscore(np.asarray(sig[:, c])) for c in range(2)]
        return [classify_row(
            rid=make_id("ltaf", rec, ws, L, "cls"), length=L, channel_names=CHANNELS,
            channel_values=vals, kind_label=KIND_LABEL[s], kind_snake=KIND_SNAKE[s],
            start=g_lo - ws, end=g_hi - ws, inj_channels=CHANNELS,
            question=classify_question(L, KIND_LABELS, DEFS), seed=int(rng.integers(1 << 31)))]
    return None


def place_absent(rng, gap, min_sec, max_sec, guard_s, retries=6):
    """Return [classify_row] for one clean (no-anomaly) window, or None."""
    guard = int(round(guard_s * HZ))
    rec, lo, hi = gap
    gap_s = (hi - lo) / HZ
    for _ in range(retries):
        L = int(round(rng.uniform(min_sec, min(max_sec, gap_s - 2 * guard_s)) * HZ))
        if hi - guard - L < lo + guard:
            continue
        ws = int(rng.integers(lo + guard, hi - guard - L + 1))
        sig = load_sig(rec)[ws:ws + L, :2]
        vals = [zscore(np.asarray(sig[:, c])) for c in range(2)]
        return [classify_row(
            rid=make_id("ltaf", rec, ws, L, "none"), length=L, channel_names=CHANNELS,
            channel_values=vals, kind_label=NONE_LABEL, kind_snake="none",
            question=classify_question(L, KIND_LABELS, DEFS), seed=int(rng.integers(1 << 31)))]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="PhysioNet ltafdb root (default: $LTAF_DATA_ROOT or ./data/ltafdb)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "datasets/ltaf-anomaly-short"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--present-frac", type=float, default=0.8)
    ap.add_argument("--min-sec", type=float, default=60.0)
    ap.add_argument("--max-sec", type=float, default=300.0)
    ap.add_argument("--guard-s", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    _bind_data_root(Path(args.data_root))
    rng = np.random.default_rng(args.seed)
    guard = int(round(args.guard_s * HZ))
    min_s = args.min_sec * HZ

    cands, gaps = [], []
    for rec in TEST:
        c, g = scan(rec)
        cands += c
        gaps += g
    # usable = corridor admits at least a min-length window with guards
    cands = [c for c in cands if (c[4] - c[3]) >= min_s + 2 * guard]
    gaps = [g for g in gaps if (g[2] - g[1]) >= min_s + 2 * guard]

    n_present = int(round(args.n * args.present_frac))
    n_absent = args.n - n_present

    # balanced V/A candidate stream (alternate, fall back when one class is dry)
    byc = {"V": [c for c in cands if c[2] == "V"], "A": [c for c in cands if c[2] == "A"]}
    for v in byc.values():
        rng.shuffle(v)
    ordered, i = [], 0
    while byc["V"] or byc["A"]:
        s = "V" if i % 2 == 0 else "A"
        if not byc[s]:
            s = "A" if s == "V" else "V"
        ordered.append(byc[s].pop())
        i += 1

    rows, n_win = [], 0
    for cand in ordered:
        if n_win >= n_present:
            break
        out = place_present(rng, cand, args.min_sec, args.max_sec, args.guard_s)
        if out:
            rows.extend(out)
            n_win += 1
    rng.shuffle(gaps)
    n_neg = 0
    for gap in gaps:
        if n_neg >= n_absent:
            break
        out = place_absent(rng, gap, args.min_sec, args.max_sec, args.guard_s)
        if out:
            rows.extend(out)
            n_neg += 1

    rng.shuffle(rows)
    write_dataset(rows, Path(args.out), {
        "dataset": "ltaf-anomaly", "domain": "LTAF ECG (real)", "source_hz": HZ,
        "anomaly": "single PVC/APC beat in NSR; gold = RR bracket",
        "records": TEST, "seed": args.seed,
    })


if __name__ == "__main__":
    main()
