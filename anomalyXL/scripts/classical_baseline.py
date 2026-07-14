#!/usr/bin/env python
"""Classical anomaly-detection baseline for AnomalyXL: detector features + trained
task-adapter heads.

One system, one row per table: per-channel classical features (robust-z
exceedances, change-points, spectral/variance/trend statistics, matrix-profile
discords, and cross-correlation for paired questions) feed small
HistGradientBoosting heads — one per question category — trained on the same
train/val splits as the TSLM baselines and evaluated once on the same test sets
with the same scorers.

The feature design encodes task understanding; the heads only learn thresholds
and combinations. Report the row as "Classical features + trained heads (SFT)".

Usage (run from environments/timeseries_qa so its .venv resolves):
  .venv/bin/python ../../anomalyXL/scripts/classical_baseline.py features
  .venv/bin/python ../../anomalyXL/scripts/classical_baseline.py run
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    import sklearn  # noqa: F401
    import stumpy
except ImportError as exc:
    raise SystemExit(
        f"classical_baseline needs the 'classical' extra (missing: {exc.name}).\n"
        "Install it with:  uv sync --extra classical"
    ) from exc

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]   # <repo>/anomalyXL/scripts/ -> <repo>
XL = REPO / "anomalyXL"
TRAIN_ROOT = XL / "datasets/anomalyxl-train"
TEST_ROOT = XL / "datasets"
CACHE = XL / "datasets/classical-cache"
ENV = str(REPO / "environments/timeseries_qa")

DEC_LEN = 16384      # decimation ceiling for stats
CP_LEN = 2048        # change-point grid
MP_LEN = 4096        # matrix-profile grid
Z_THR = 4.0

KINDS = ["No Anomaly", "Level Shift", "Transient Spike", "Change in Seasonality",
         "Change in Variance", "Change in Trend"]
MAG_BINS = ["No Anomaly", "Smallest", "Small", "Medium", "Large"]
ST_BINS = ["No Anomaly", "Before", "Early", "Medium", "Late"]
IND_CLASSES = ["no_anomaly", "correlated", "independent", "lead", "lag"]
IND_KEYWORDS = {  # option-text keyword -> class
    "Perfectly correlated": "correlated", "Not correlated": "independent",
    "Leading indicator": "lead", "Lagging indicator": "lag", "No anomaly": "no_anomaly",
}


# --------------------------------------------------------------------- features

def _dec(x: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    stride = max(1, len(x) // target)
    return x[::stride], stride


def channel_features(x: np.ndarray) -> dict:
    """~30 classical statistics for one channel."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0)
    L = len(x)
    xd, stride = _dec(x, DEC_LEN)
    m = len(xd)
    med = np.median(xd)
    mad = np.median(np.abs(xd - med)) * 1.4826 + 1e-9
    z = (xd - med) / mad
    az = np.abs(z)
    amax = float(az.max())
    argmax_f = float(az.argmax() / m)
    exc = az > Z_THR
    f = {"amax": amax, "argmax_f": argmax_f,
         "frac4": float(exc.mean()), "frac8": float((az > 8).mean())}
    # exceedance runs
    if exc.any():
        idx = np.flatnonzero(exc)
        gaps = np.flatnonzero(np.diff(idx) > max(3, m // 512))
        starts = np.r_[idx[0], idx[gaps + 1]]
        ends = np.r_[idx[gaps], idx[-1]]
        lens = ends - starts + 1
        k = int(lens.argmax())
        f.update(exc_first_f=float(starts[0] / m), exc_last_f=float((ends[-1] + 1) / m),
                 run_start_f=float(starts[k] / m), run_end_f=float((ends[k] + 1) / m),
                 run_len_f=float(lens[k] / m), n_runs=float(len(starts)))
    else:
        f.update(exc_first_f=1.0, exc_last_f=1.0, run_start_f=1.0, run_end_f=1.0,
                 run_len_f=0.0, n_runs=0.0)
    # spike width at |z|>6 around the peak
    j = int(az.argmax()); lo = j; hi = j
    while lo > 0 and az[lo - 1] > 6: lo -= 1
    while hi < m - 1 and az[hi + 1] > 6: hi += 1
    f["peak_width_f"] = float((hi - lo + 1) / m)
    # halves: level shift / variance change / trend
    h1, h2 = xd[: m // 2], xd[m // 2:]
    sall = xd.std() + 1e-9
    f["half_shift"] = float((h2.mean() - h1.mean()) / sall)
    f["half_logvar"] = float(np.log((h2.std() + 1e-9) / (h1.std() + 1e-9)))
    t = np.linspace(0, 1, m)
    f["slope"] = float(np.polyfit(t, (xd - med) / sall, 1)[0])
    # rolling variance localization (variance-change onset)
    w = max(8, m // 64)
    rv = pd.Series(xd).rolling(w).std().to_numpy()
    rv = np.nan_to_num(rv, nan=0.0)
    rvz = (rv - np.median(rv)) / (np.median(np.abs(rv - np.median(rv))) * 1.4826 + 1e-9)
    f["rv_max"] = float(np.abs(rvz).max())
    f["rv_arg_f"] = float(np.abs(rvz).argmax() / m)
    # spectral: dominant-frequency power ratio between halves
    def top_pow(seg):
        if len(seg) < 16: return 0.0, 0.0
        p = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
        p[0] = 0
        return float(p.max() / (p.sum() + 1e-9)), float(p.argmax() / len(p))
    p1, f1 = top_pow(h1); p2, f2 = top_pow(h2)
    f["spec_pow_d"] = p2 - p1
    f["spec_frq_d"] = f2 - f1
    # change points (PELT on coarse grid)
    try:
        xc, _ = _dec(xd, CP_LEN)
        xn = (xc - med) / sall
        bkps = rpt.Pelt(model="l2", min_size=16, jump=4).fit(xn).predict(pen=10.0)
        bkps = [b for b in bkps if b < len(xc)]
        f["n_cp"] = float(len(bkps))
        if bkps:
            segs = np.split(xn, bkps)
            means = [s.mean() for s in segs]
            jumps = np.abs(np.diff(means))
            k = int(jumps.argmax())
            f["cp_first_f"] = float(bkps[0] / len(xc))
            f["cp_maxjump"] = float(jumps.max()) if len(jumps) else 0.0
            f["cp_maxjump_f"] = float(bkps[k] / len(xc)) if len(jumps) else 1.0
        else:
            f.update(cp_first_f=1.0, cp_maxjump=0.0, cp_maxjump_f=1.0)
    except Exception:
        f.update(n_cp=0.0, cp_first_f=1.0, cp_maxjump=0.0, cp_maxjump_f=1.0)
    # matrix profile discord
    try:
        xm, _ = _dec(xd, MP_LEN)
        wmp = max(32, len(xm) // 64)
        mp = stumpy.stump((xm - med) / sall, m=wmp)[:, 0].astype(float)
        f["mp_max"] = float(np.nanmax(mp))
        f["mp_arg_f"] = float(np.nanargmax(mp) / len(mp))
    except Exception:
        f.update(mp_max=0.0, mp_arg_f=1.0)
    f["log_len"] = float(np.log2(L))
    return f


def zscore_series(x: np.ndarray, target: int = CP_LEN) -> np.ndarray:
    xd, _ = _dec(np.nan_to_num(np.asarray(x, float)), target)
    med = np.median(xd)
    mad = np.median(np.abs(xd - med)) * 1.4826 + 1e-9
    return np.abs((xd - med) / mad)


def question_features(row: dict) -> dict:
    chans = [str(c) for c in row["channel_names"]]
    vals = [np.asarray(v, dtype=np.float32) for v in row["channel_values"]]
    per = {c: channel_features(v) for c, v in zip(chans, vals)}
    keys = list(next(iter(per.values())).keys())
    # question-level aggregate: worst channel (by amax) + mean
    worst = max(per, key=lambda c: per[c]["amax"])
    agg = {f"w_{k}": per[worst][k] for k in keys}
    agg.update({f"m_{k}": float(np.mean([per[c][k] for c in per])) for k in keys})
    agg["n_channels"] = float(len(chans))
    # paired A_/B_ cross-correlation (indicator / lead_lag)
    A = [v for c, v in zip(chans, vals) if c.startswith("A_")]
    B = [v for c, v in zip(chans, vals) if c.startswith("B_")]
    if A and B:
        za = np.mean([zscore_series(v) for v in A], axis=0)
        zb = np.mean([zscore_series(v) for v in B], axis=0)
        za = (za - za.mean()) / (za.std() + 1e-9)
        zb = (zb - zb.mean()) / (zb.std() + 1e-9)
        n = len(za)
        cc = np.correlate(za, zb, mode="full") / n
        lags = np.arange(-n + 1, n)
        keep = np.abs(lags) <= n // 2
        cc, lags = cc[keep], lags[keep]
        k = int(np.abs(cc).argmax())
        agg.update(xc_best=float(cc[k]), xc_lag_f=float(lags[k] / n),
                   xc_zero=float(cc[lags == 0][0]),
                   a_amax=float(max(per[c]["amax"] for c in per if c.startswith("A_"))),
                   b_amax=float(max(per[c]["amax"] for c in per if c.startswith("B_"))))
    else:
        agg.update(xc_best=0.0, xc_lag_f=0.0, xc_zero=0.0, a_amax=0.0, b_amax=0.0)
    return {"agg": agg, "per": per}


def _worker(args):
    i, row = args
    try:
        return i, question_features(row)
    except Exception as e:
        return i, {"error": str(e)}


def build_features(variant: str, split: str) -> None:
    if split == "test":
        src = TEST_ROOT / f"anomalyxl-{variant}/data.parquet"
    else:
        src = TRAIN_ROOT / f"anomalyxl-{variant}/{split}.parquet"
    df = pd.read_parquet(src)
    out_p = CACHE / f"{variant}-{split}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rows = [(i, dict(r)) for i, (_, r) in enumerate(df.iterrows())]
    results: dict[int, dict] = {}
    with Pool(min(32, len(rows))) as pool:
        for n, (i, feat) in enumerate(pool.imap_unordered(_worker, rows), 1):
            results[i] = feat
            if n % 200 == 0:
                print(f"  {variant}-{split}: {n}/{len(rows)}", flush=True)
    payload = []
    for i in range(len(df)):
        r = df.iloc[i]
        payload.append({"id": r["id"], "category": r["category"], "length": int(r["length"]),
                        "channel_names": [str(c) for c in r["channel_names"]],
                        "question": r["question"],
                        "options": [str(o) for o in r["options"]],
                        "answer": str(r["answer"]), **results[i]})
    out_p.write_text(json.dumps(payload))
    print(f"wrote {out_p} ({len(payload)} rows)", flush=True)


# ---------------------------------------------------------------- heads + eval

def load_feats(variant, split):
    return json.loads((CACHE / f"{variant}-{split}.json").read_text())


def vec(agg: dict, names: list[str]) -> list[float]:
    return [float(agg.get(k, 0.0)) for k in names]


def listed_channels(question: str, chans: list[str]) -> list[str]:
    hits = re.findall(r"ch_\d+", question)
    return [c for c in dict.fromkeys(hits) if c in chans] or chans


def train_eval():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGC
    from sklearn.ensemble import HistGradientBoostingRegressor as HGR
    sys.path.insert(0, ENV)
    from ts_datasets.anomalyxl import AnomalyXL

    def fit_clf(X, y):
        m = HGC(max_iter=300, early_stopping=True)
        m.fit(np.array(X), np.array(y)); return m

    def fit_reg(X, y):
        m = HGR(max_iter=300, early_stopping=True)
        m.fit(np.array(X), np.array(y)); return m

    report = {}
    for variant in ("coarse", "precise"):
        tr = load_feats(variant, "train") + load_feats(variant, "val")
        te = load_feats(variant, "test")
        names = sorted(tr[0]["agg"].keys())
        by_cat = {}
        for r in tr:
            by_cat.setdefault(r["category"], []).append(r)
        heads = {}

        def gold(r):
            a = r["answer"]
            return json.loads(a.split(":", 1)[1]) if variant == "precise" else a

        for cat, rows in by_cat.items():
            X = [vec(r["agg"], names) for r in rows]
            if variant == "coarse":
                if cat == "presence":
                    heads[cat] = fit_clf(X, [r["answer"] for r in rows])
                elif cat == "start_time":
                    lab = ["No Anomaly" if r["answer"] == "No Anomaly" else r["answer"].split(" ")[0] for r in rows]
                    heads[cat] = fit_clf(X, lab)
                elif cat in ("categorization", "magnitude"):
                    heads[cat] = fit_clf(X, [r["answer"] for r in rows])
                elif cat == "indicator":
                    lab = []
                    for r in rows:
                        lab.append(next(v for k, v in IND_KEYWORDS.items() if k in r["answer"]))
                    heads[cat] = fit_clf(X, lab)
                elif cat == "identification":
                    Xc, yc = [], []
                    for r in rows:
                        gold_set = set() if r["answer"] == "None" else {s.strip() for s in r["answer"].split(",")}
                        for c in listed_channels(r["question"], r["channel_names"]):
                            Xc.append(vec(r["per"][c], sorted(r["per"][c].keys())) + vec(r["agg"], names))
                            yc.append(int(c in gold_set))
                    heads[cat] = fit_clf(Xc, yc)
            else:
                if cat == "localize":
                    heads["loc_pres"] = fit_clf(X, [int(gold(r)["present"]) for r in rows])
                    pres = [(x, gold(r)) for x, r in zip(X, rows) if gold(r)["present"]]
                    heads["loc_start"] = fit_reg([x for x, g in pres], [g["start"] / g["_L"] for x, g in pres])
                    heads["loc_end"] = fit_reg([x for x, g in pres], [g["end"] / g["_L"] for x, g in pres])
                elif cat == "classify_with_evidence":
                    heads["cwe_kind"] = fit_clf(X, [gold(r)["kind"] for r in rows])
                    heads["cwe_start"] = fit_reg(X, [gold(r)["start"] / gold(r)["_L"] for r in rows])
                    heads["cwe_end"] = fit_reg(X, [gold(r)["end"] / gold(r)["_L"] for r in rows])
                elif cat == "measure_magnitude":
                    heads["mm"] = fit_reg(X, [np.log1p(gold(r)["magnitude_sigma"]) for r in rows])
                elif cat == "localize_all_channels":
                    Xc, yc = [], []
                    for r in rows:
                        gold_ch = {a["channel"] for a in gold(r)["anomalies"]}
                        for c in r["channel_names"]:
                            Xc.append(vec(r["per"][c], sorted(r["per"][c].keys())) + vec(r["agg"], names))
                            yc.append(int(c in gold_ch))
                    heads["lac"] = fit_clf(Xc, yc)
                elif cat == "lead_lag_with_magnitude":
                    heads["ll_dir"] = fit_clf(X, [gold(r)["direction"] for r in rows])
                    dd = [(x, gold(r)) for x, r in zip(X, rows) if gold(r)["direction"] in ("lead", "lag")]
                    heads["ll_lag"] = fit_reg([x for x, g in dd], [abs(g["lag_samples"]) / g["_L"] for x, g in dd])

        # ---- predict on test
        ds = AnomalyXL(repo_or_path=str(TEST_ROOT / f"anomalyxl-{variant}"))
        preds, golds, cats = [], [], []
        for r in te:
            x = np.array([vec(r["agg"], names)])
            cat, L = r["category"], r["length"]
            if variant == "coarse":
                if cat == "presence":
                    p = heads[cat].predict(x)[0]
                elif cat == "start_time":
                    lab = heads[cat].predict(x)[0]
                    p = "No Anomaly" if lab == "No Anomaly" else next(
                        (o for o in r["options"] if o.startswith(lab)), r["options"][0])
                elif cat in ("categorization", "magnitude"):
                    p = heads[cat].predict(x)[0]
                    if p not in r["options"]:
                        p = next((o for o in r["options"] if o == p), r["options"][0])
                elif cat == "indicator":
                    lab = heads[cat].predict(x)[0]
                    p = next((o for o in r["options"] for k, v in IND_KEYWORDS.items()
                              if v == lab and k in o), r["options"][0])
                elif cat == "identification":
                    sel = []
                    for c in listed_channels(r["question"], r["channel_names"]):
                        xc = np.array([vec(r["per"][c], sorted(r["per"][c].keys())) + vec(r["agg"], names)])
                        if heads[cat].predict(xc)[0] == 1:
                            sel.append(c)
                    want = ", ".join(sorted(sel)) if sel else "None"
                    p = next((o for o in r["options"]
                              if ({s.strip() for s in o.split(",")} if o != "None" else set()) == set(sel)), want)
            else:
                if cat == "localize":
                    pres = bool(heads["loc_pres"].predict(x)[0])
                    obj = {"_L": L, "present": pres}
                    if pres:
                        s = float(np.clip(heads["loc_start"].predict(x)[0], 0, 1))
                        e = float(np.clip(heads["loc_end"].predict(x)[0], 0, 1))
                        obj["start"], obj["end"] = int(s * L), int(max(e, s + 1e-4) * L)
                    p = json.dumps(obj)
                elif cat == "classify_with_evidence":
                    s = float(np.clip(heads["cwe_start"].predict(x)[0], 0, 1))
                    e = float(np.clip(heads["cwe_end"].predict(x)[0], 0, 1))
                    p = json.dumps({"_L": L, "kind": heads["cwe_kind"].predict(x)[0],
                                    "start": int(s * L), "end": int(max(e, s + 1e-4) * L)})
                elif cat == "measure_magnitude":
                    p = json.dumps({"_L": L, "magnitude_sigma": float(np.expm1(heads["mm"].predict(x)[0]))})
                elif cat == "localize_all_channels":
                    anoms = []
                    for c in r["channel_names"]:
                        xc = np.array([vec(r["per"][c], sorted(r["per"][c].keys())) + vec(r["agg"], names)])
                        if heads["lac"].predict(xc)[0] == 1:
                            pf = r["per"][c]
                            anoms.append({"channel": c, "start": int(pf["run_start_f"] * L),
                                          "end": int(max(pf["run_end_f"], pf["run_start_f"] + 1e-4) * L)})
                    p = json.dumps({"_L": L, "anomalies": anoms})
                elif cat == "lead_lag_with_magnitude":
                    d = heads["ll_dir"].predict(x)[0]
                    obj = {"_L": L, "direction": d}
                    if d in ("lead", "lag"):
                        obj["lag_samples"] = int(float(np.clip(heads["ll_lag"].predict(x)[0], 0, 1)) * L)
                    p = json.dumps(obj)
            preds.append(p); golds.append(r["answer"]); cats.append(cat)

        # Coarse golds are plain option text (exact match, as in the release
        # evals); the AnomalyXL scorer handles the precise structured format.
        if variant == "coarse":
            scores = [float(str(p).strip() == str(g).strip()) for p, g in zip(preds, golds)]
        else:
            scores = [ds.score_answer(p, g) for p, g in zip(preds, golds)]
        sdf = pd.DataFrame({"category": cats, "score": scores, "pred": preds, "gold": golds})
        report[variant] = sdf
        print(f"\n===== CLASSICAL+HEADS on {variant}: {np.mean(scores):.4f} =====")
        print(sdf.groupby("category")["score"].agg(["mean", "count"]).to_string())
        sdf.to_csv(CACHE / f"classical-{variant}-test-scored.csv", index=False)
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "features":
        for variant in ("coarse", "precise"):
            for split in ("train", "val", "test"):
                if not (CACHE / f"{variant}-{split}.json").exists():
                    build_features(variant, split)
    else:
        train_eval()
