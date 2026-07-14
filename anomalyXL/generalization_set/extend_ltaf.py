"""Stage 2 of the LTAF generalization set: the LONG-context windows.

`gen_ltaf.py` (stage 1) draws the short windows (60-300 s). This script appends the
long ones (>= 300 s) with strict per-class (V/A) quotas, excluding any id already
drawn, and writes the merged 250-row set the environment loads. The stage-1 rows are
carried over byte-identical, and a separate seed keeps the long draw independent of
the short one.

  Stage 1 rows are read from --existing; the merged set is written to --out.
  gen_ltaf.py must be importable (module-level place_present/place_absent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import gen_ltaf as G
from common import _COLUMNS, NONE_LABEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default=str(Path(__file__).parent / "datasets/ltaf-anomaly-short/data.parquet"),
                    help="stage-1 (short-window) parquet from gen_ltaf.py")
    ap.add_argument("--out", default=str(Path(__file__).parent / "datasets/ltaf-anomaly"),
                    help="write dir for the merged set (stage-1 dir left intact)")
    ap.add_argument("--n-per-class", type=int, default=60, help="positives per V and A")
    ap.add_argument("--n-neg", type=int, default=30)
    ap.add_argument("--min-sec", type=float, default=300.0)
    ap.add_argument("--max-sec", type=float, default=1024.0)  # 1024*128 = 131072 samples
    ap.add_argument("--guard-s", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--write", action="store_true", help="write the merged parquet (else dry-run: report yield only)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    guard = int(round(args.guard_s * G.HZ))
    floor_samples = args.min_sec * G.HZ + 2 * guard

    existing = pd.read_parquet(args.existing)
    used_ids = set(existing["id"])

    cands, gaps = [], []
    for rec in G.TEST:
        c, g = G.scan(rec)
        cands += c
        gaps += g
    byc = {"V": [c for c in cands if c[2] == "V" and (c[4] - c[3]) >= floor_samples],
           "A": [c for c in cands if c[2] == "A" and (c[4] - c[3]) >= floor_samples]}
    gaps = [g for g in gaps if (g[2] - g[1]) >= floor_samples]
    for v in byc.values():
        rng.shuffle(v)
    rng.shuffle(gaps)

    def take(pool, quota, placer):
        out = []
        for item in pool:
            if len(out) >= quota:
                break
            r = placer(rng, item, args.min_sec, args.max_sec, args.guard_s)
            if r and r[0]["id"] not in used_ids:
                used_ids.add(r[0]["id"])
                out.extend(r)
        return out

    v_rows = take(byc["V"], args.n_per_class, G.place_present)
    a_rows = take(byc["A"], args.n_per_class, G.place_present)
    neg_rows = take(gaps, args.n_neg, G.place_absent)

    print(f"pool usable>= {args.min_sec:.0f}s: V={len(byc['V'])} A={len(byc['A'])} neg={len(gaps)}")
    print(f"placed: V={len(v_rows)}/{args.n_per_class}  A={len(a_rows)}/{args.n_per_class}  "
          f"neg={len(neg_rows)}/{args.n_neg}")
    ok = len(v_rows) == args.n_per_class and len(a_rows) == args.n_per_class and len(neg_rows) == args.n_neg
    print(f"QUOTAS MET: {ok}")
    if v_rows:
        sec = np.array([r["length"] / G.HZ for r in v_rows + a_rows])
        print(f"new-positive length (s): min {sec.min():.0f} / med {np.median(sec):.0f} / max {sec.max():.0f}")

    if not (args.write and ok):
        if args.write and not ok:
            print("NOT writing: quotas unmet.")
        return

    new = pd.DataFrame(v_rows + a_rows + neg_rows, columns=_COLUMNS)
    merged = pd.concat([existing, new], ignore_index=True)
    assert list(merged["id"][:len(existing)]) == list(existing["id"]), "existing rows perturbed!"
    assert merged["id"].is_unique, "duplicate ids after merge!"

    out_dir = Path(args.out) if args.out else Path(args.existing).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "data.parquet")
    n_none = int(merged["answer"].str.contains(f'"kind": "{NONE_LABEL}"').sum())
    manifest = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").exists() else {}
    manifest.update({
        "dataset": "ltaf-anomaly", "domain": "LTAF ECG (real)", "source_hz": G.HZ,
        "anomaly": "single PVC/APC beat in NSR; gold = RR bracket",
        "records": G.TEST, "n_rows": len(merged), "n_anomaly": len(merged) - n_none,
        "n_no_anomaly": n_none,
        "window_seconds": {"short": [60.0, 300.0], "long": [args.min_sec, args.max_sec]},
        "seeds": {"short_windows": 42, "long_windows": args.seed},
        "long_windows": {"n_per_class": args.n_per_class, "n_neg": args.n_neg},
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir/'data.parquet'}: {len(merged)} rows "
          f"({len(merged)-n_none} anomaly / {n_none} no-anomaly)")


if __name__ == "__main__":
    main()
