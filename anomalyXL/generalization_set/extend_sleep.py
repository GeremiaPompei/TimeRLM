"""Stage 2 of the Sleep-PSG generalization set: the LONG-context windows.

`gen_sleep.py` (stage 1) draws the short windows (60-300 s). This script appends the
long ones (>= 300 s) with strict per-activity quotas, excluding any id already drawn,
and writes the merged 250-row set the environment loads. The stage-1 rows are carried
over byte-identical, and a separate seed keeps the long draw independent of the short
one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import gen_sleep as G
from common import _COLUMNS, NONE_LABEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default=str(Path(__file__).parent / "datasets/sleep-anomaly-short/data.parquet"),
                    help="stage-1 (short-window) parquet from gen_sleep.py")
    ap.add_argument("--out", default=str(Path(__file__).parent / "datasets/sleep-anomaly"),
                    help="write dir for the merged set (stage-1 dir left intact)")
    ap.add_argument("--n-per-class", type=int, default=30, help="positives per activity class")
    ap.add_argument("--n-neg", type=int, default=30)
    ap.add_argument("--min-sec", type=float, default=300.0)
    ap.add_argument("--max-sec", type=float, default=655.0)  # 655*200 = 131000 samples
    ap.add_argument("--guard-s", type=float, default=5.0)
    ap.add_argument("--channels", default=",".join(G.DEFAULT_CHANNELS))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--write", action="store_true", help="write the merged parquet (else dry-run: report yield only)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sel = [c.strip() for c in args.channels.split(",") if c.strip()]
    test = json.loads(G.SPLIT.read_text())["test"]

    existing = pd.read_parquet(args.existing)
    used_ids = set(existing["id"])

    cands, gaps = [], []
    for rec in test:
        if not (G.AROUSAL / f"{rec}.parquet").exists():
            continue
        c, g = G.scan(rec)
        cands += c
        gaps += g

    # usable present at this floor: window range non-empty (holds event + fits corridor)
    def usable(c):
        dur_s, corr_s = (c[2] - c[1]) / G.HZ, (c[5] - c[4]) / G.HZ
        a, b = G.lo_hi_sec(dur_s, corr_s, args.min_sec, args.max_sec, args.guard_s)
        return a <= b

    bya = {k: [c for c in cands if c[3] == k and usable(c)] for k in sorted(G.RESP)}
    gaps = [g for g in gaps if (g[2] - g[1]) / G.HZ >= args.min_sec + 2 * args.guard_s]
    for v in bya.values():
        rng.shuffle(v)
    rng.shuffle(gaps)

    def take(pool, quota, placer):
        out = []
        for item in pool:
            if len(out) >= quota:
                break
            r = placer(rng, item, args.min_sec, args.max_sec, args.guard_s, sel)
            if r and r[0]["id"] not in used_ids:
                used_ids.add(r[0]["id"])
                out.extend(r)
        return out

    pos_rows, placed = [], {}
    for k in sorted(G.RESP):
        rows_k = take(bya[k], args.n_per_class, G.place_present)
        placed[k] = len(rows_k)
        pos_rows += rows_k
    neg_rows = take(gaps, args.n_neg, G.place_absent)

    print("pool usable>= {:.0f}s: {} neg={}".format(
        args.min_sec, {k: len(v) for k, v in bya.items()}, len(gaps)))
    print(f"placed per class (target {args.n_per_class}): {placed}   neg={len(neg_rows)}/{args.n_neg}")
    ok = all(placed[k] == args.n_per_class for k in G.RESP) and len(neg_rows) == args.n_neg
    print(f"QUOTAS MET: {ok}")
    if pos_rows:
        sec = np.array([r["length"] / G.HZ for r in pos_rows])
        print(f"new-positive length (s): min {sec.min():.0f} / med {np.median(sec):.0f} / max {sec.max():.0f}")

    if not (args.write and ok):
        if args.write and not ok:
            print("NOT writing: quotas unmet.")
        return

    new = pd.DataFrame(pos_rows + neg_rows, columns=_COLUMNS)
    merged = pd.concat([existing, new], ignore_index=True)
    assert list(merged["id"][:len(existing)]) == list(existing["id"]), "existing rows perturbed!"
    assert merged["id"].is_unique, "duplicate ids after merge!"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "data.parquet")
    n_none = int(merged["answer"].str.contains(f'"kind": "{NONE_LABEL}"').sum())
    (out_dir / "manifest.json").write_text(json.dumps({
        "dataset": "sleep-anomaly", "domain": "Sleep-PSG (real)", "source_hz": G.HZ,
        "anomaly": "single respiratory arousal bout; gold = native bout range",
        "channels": sel, "n_test_records": len(test), "n_rows": len(merged),
        "n_anomaly": len(merged) - n_none, "n_no_anomaly": n_none,
        "window_seconds": {"short": [60.0, 300.0], "long": [args.min_sec, args.max_sec]},
        "seeds": {"short_windows": 42, "long_windows": args.seed},
        "long_windows": {"n_per_class": args.n_per_class, "n_neg": args.n_neg},
    }, indent=2))
    print(f"wrote {out_dir/'data.parquet'}: {len(merged)} rows "
          f"({len(merged)-n_none} anomaly / {n_none} no-anomaly)")


if __name__ == "__main__":
    main()
