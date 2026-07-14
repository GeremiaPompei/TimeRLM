"""Per-dataset / per-category scorecard for the generalization-set evals.

`vf-eval` reports each metric averaged over *all* rows, which dilutes the
classify metrics with the no-anomaly negatives (and vice versa). This regroups a
run's `results.jsonl` by category — `anomaly` (classify_with_evidence) vs
`no_anomaly` (localize present=false) — and by anomaly kind, and reports the
three signals separately plus the combined one:

  classification = mean(kind_correct)      — did it name the right anomaly?
  IoU            = mean(iou)               — did it localize the window?
  combined       = mean(kind_x_iou)        — kind AND window (the headline)
  no_anomaly     = mean(presence_correct)  — did it correctly say "none"?

Usage (one or more runs; dirs are searched for results.jsonl):
  python scripts/report_generalization.py outputs/evals/generalization/**/results.jsonl
  python scripts/report_generalization.py outputs/evals/generalization --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


_KIND_RE = re.compile(r'kind["\']?\s*[:=]\s*["\']([^"\']+)["\']')


def _pred_kind_from_text(s: str | None) -> str | None:
    """Last `kind: "..."` mention in a blob (the model's final answer wins)."""
    if not s:
        return None
    m = _KIND_RE.findall(s)
    return m[-1] if m else None


def _rlm_pred_kind(row: dict) -> str | None:
    """Recover the predicted kind from a results.jsonl row. RLM answers live in the
    REPL `answer['content']=...` code (assistant tool_calls); single-pass answers
    live in the assistant message content. Scan both; the last `kind` wins."""
    codes: list[str] = []      # REPL answer['content']=... (RLM final answer)
    contents: list[str] = []   # plain assistant text (single-pass final answer)
    for m in row.get("completion") or []:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            contents.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    contents.append(str(part.get("text", "")))
        for tc in m.get("tool_calls") or []:
            try:
                call = json.loads(tc) if isinstance(tc, str) else tc
                args = call.get("arguments") or (call.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                code = args.get("code") if isinstance(args, dict) else None
                if code:
                    codes.append(str(code))
            except (json.JSONDecodeError, AttributeError):
                pass
    # Prefer the REPL-submitted answer (RLM); fall back to assistant text (single-pass).
    return _pred_kind_from_text("\n".join(codes)) or _pred_kind_from_text("\n".join(contents))


def _rows(path: Path) -> list[dict]:
    """Normalize either harness's output to rows of {answer, metrics, reward, pred_kind}:
    vf-eval `results.jsonl` (one JSON per line) or a baseline harness's `eval_test.json`
    (a dict with an `outputs` list of per-sample entries)."""
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["pred_kind"] = _rlm_pred_kind(r)
            rows.append(r)
        return rows
    doc = json.loads(path.read_text())
    rows = []
    for o in doc.get("outputs", []):
        rows.append({
            "answer": o.get("ground_truth", ""),
            "reward": float(o.get("primary", 0.0)),
            "metrics": {k: o[k] for k in ("kind_correct", "iou", "kind_x_iou",
                                          "kind_and_iou_above_0p5", "acc_at_iou_0p5")
                        if o.get(k) is not None},
            "pred_kind": _pred_kind_from_text(o.get("predicted_answer", "")),
        })
    return rows


def _label(path: Path) -> str:
    """Best-effort run label: nearest ancestor dir naming the dataset/cell."""
    for part in reversed(path.parts):
        low = part.lower()
        if "ltaf" in low or "sleep" in low or "anomaly" in low or "generaliz" in low:
            return part
    return path.parent.name


NONE_LABEL = "No anomaly"


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro-averaged F1 over the K anomaly *kinds* (the "No anomaly" class is
    excluded so abstention precision can't inflate the score). All rows still
    count, so predicting a kind on a clean window is a false positive. Any
    collapse -- to one kind, or to "No anomaly" -- drives this toward 0."""
    classes = sorted({g for g, _ in pairs if g != NONE_LABEL})
    if not classes:
        return float("nan")
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return _mean(f1s)


def score_run(path: Path) -> dict:
    anomaly, no_anom_correct, by_kind, rewards = [], [], {}, []
    pairs: list[tuple[str, str]] = []   # (gold_kind, pred_kind) over all classify rows
    raw_preds: list[str] = []           # recovered predictions, for collapse rate
    for r in _rows(path):
        ans = str(r.get("answer", ""))
        prefix, _, payload = ans.partition(":")
        m = r.get("metrics", {})
        rewards.append(float(r.get("reward", 0.0)))
        try:
            kind = json.loads(payload).get("kind", "?")
        except json.JSONDecodeError:
            kind = "?"
        if prefix == "localize":  # legacy split-prompt negatives
            no_anom_correct.append(float(r.get("reward", 0.0)))
            continue
        # unified classify schema (anomaly kinds + "No anomaly")
        if kind == NONE_LABEL:
            no_anom_correct.append(float(m.get("kind_correct", r.get("reward", 0.0))))
        else:
            anomaly.append(m)
            by_kind.setdefault(kind, []).append(m)
        # confusion pair: diagonal is exact (kind_correct); off-diagonal uses the
        # recovered prediction (validated to disagree with gold on wrong rows).
        kc = m.get("kind_correct")
        pk = r.get("pred_kind")
        pred = kind if (kc is not None and float(kc) >= 1.0) else (pk or "(none)")
        pairs.append((kind, pred))
        raw_preds.append(pk or "(none)")
    collapse = (max(Counter(raw_preds).values()) / len(raw_preds)) if raw_preds else float("nan")
    return {
        "label": _label(path),
        "n_anomaly": len(anomaly),
        "n_no_anomaly": len(no_anom_correct),
        "classification": _mean([m.get("kind_correct", 0.0) for m in anomaly]),
        "macro_f1": _macro_f1(pairs),
        "collapse_rate": collapse,
        "iou": _mean([m.get("iou", 0.0) for m in anomaly]),
        "combined": _mean([m.get("kind_x_iou", 0.0) for m in anomaly]),
        "acc_at_iou_0p5": _mean([m.get("acc_at_iou_0p5", 0.0) for m in anomaly]),
        "no_anomaly_correct": _mean(no_anom_correct),  # did it answer "No anomaly"?
        "overall_reward": _mean(rewards),
        "by_kind": {
            k: {
                "n": len(ms),
                "classification": _mean([m.get("kind_correct", 0.0) for m in ms]),
                "iou": _mean([m.get("iou", 0.0) for m in ms]),
                "combined": _mean([m.get("kind_x_iou", 0.0) for m in ms]),
            }
            for k, ms in sorted(by_kind.items())
        },
    }


def _fmt(s: dict) -> str:
    L = [f"\n### {s['label']}  (anomaly={s['n_anomaly']}, no_anomaly={s['n_no_anomaly']})"]
    L.append(f"  classification={s['classification']:.3f}  IoU={s['iou']:.3f}  "
             f"combined(kind x IoU)={s['combined']:.3f}  acc@IoU0.5={s['acc_at_iou_0p5']:.3f}")
    L.append(f"  no_anomaly_correct={s['no_anomaly_correct']:.3f}   overall_reward={s['overall_reward']:.3f}")
    if s["by_kind"]:
        L.append("  per-kind:")
        for k, v in s["by_kind"].items():
            L.append(f"    {k:<42} n={v['n']:<3} class={v['classification']:.3f} "
                     f"IoU={v['iou']:.3f} combined={v['combined']:.3f}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="results.jsonl files or dirs to search")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(pp.rglob("results.jsonl")) + sorted(pp.rglob("eval_test.json"))
        elif pp.exists():
            files.append(pp)
    if not files:
        raise SystemExit("no results.jsonl found in: " + ", ".join(args.paths))

    scores = [score_run(f) for f in files]
    if args.json:
        print(json.dumps(scores, indent=2))
    else:
        print("# Generalization scorecard")
        for s in scores:
            print(_fmt(s))


if __name__ == "__main__":
    main()
