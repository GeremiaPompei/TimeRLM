"""The three AnomalyXL scorers must agree.

The paper reports TimeRLM, the TSLM baselines, the Toto baseline and the classical
baseline on one table, scored by the same metrics. In practice each of the three
subprojects carries its own copy of the scorer, because each has its own venv:

    environments/timeseries_qa/ts_datasets/anomalyxl.py   (canonical)
    sft/tslm/src/datasets/anomaly_xl/scoring.py           (vendored)
    sft/toto/src/utils/anomalyxl_scoring.py               (vendored)

If the copies drift, the table silently compares systems on different metrics. This
test scores the same answers with all three and asserts they agree, so a drift shows
up here rather than in a number nobody can reproduce.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from ts_datasets.anomalyxl import AnomalyXL

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED = {
    "tslm": _REPO_ROOT / "sft/tslm/src/datasets/anomaly_xl/scoring.py",
    "toto": _REPO_ROOT / "sft/toto/src/utils/anomalyxl_scoring.py",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(pred: str, gold: str) -> float:
    return AnomalyXL.score_answer(AnomalyXL.__new__(AnomalyXL), pred, gold)


def _cases(n: int = 400) -> list[tuple[str, str]]:
    """Random (pred, gold) pairs over all five categories, including the malformed
    and no-anomaly answers that used to raise out of the scorer."""
    rng = random.Random(0)
    out = []
    for _ in range(n):
        L = rng.choice([1000, 16384, 131072])
        cat = rng.choice([
            "localize", "classify_with_evidence", "measure_magnitude",
            "localize_all_channels", "lead_lag_with_magnitude",
        ])
        if cat == "localize":
            present = rng.random() < 0.8
            s = rng.randrange(L - 300)
            gold = {"present": present, "_L": L}
            if present:
                gold |= {"start": s, "end": s + rng.randrange(8, 256)}
            pred = rng.choice([
                {"present": True, "start": rng.randrange(L - 300), "end": rng.randrange(L)},
                {"present": False},
                {"present": True, "start": None, "end": None},
                {},
            ])
        elif cat == "classify_with_evidence":
            kind = rng.choice(["Level Shift", "Transient Spike", "No anomaly"])
            s = rng.randrange(L - 300)
            gold = {"kind": kind, "_L": L}
            if kind != "No anomaly":
                gold |= {"start": s, "end": s + 50}
            pred = rng.choice([
                {"kind": kind, "start": rng.randrange(L - 300), "end": rng.randrange(L)},
                {"kind": "No anomaly"},
                {"kind": kind, "start": None, "end": None},
            ])
        elif cat == "measure_magnitude":
            gold = {"magnitude_sigma": round(rng.uniform(1, 100), 2), "_L": L}
            pred = rng.choice([
                {"magnitude_sigma": round(rng.uniform(1, 100), 2)},
                {"magnitude_sigma": None},
                {},
            ])
        elif cat == "localize_all_channels":
            def events(k):
                return [
                    {"channel": f"ch_{i:02d}", "start": (s := rng.randrange(L - 300)), "end": s + 60}
                    for i in range(k)
                ]
            gold = {"anomalies": events(rng.randrange(0, 4)), "_L": L}
            pred = {"anomalies": events(rng.randrange(0, 4))}
        else:
            gold = {
                "direction": rng.choice(["lead", "lag", "independent"]),
                "lag_samples": rng.randrange(0, L // 4),
                "_L": L,
            }
            pred = rng.choice([
                {"direction": rng.choice(["lead", "lag", "independent"]),
                 "lag_samples": rng.randrange(0, L // 4)},
                {"direction": "lead", "lag_samples": None},
            ])
        out.append((json.dumps(pred), f"{cat}:" + json.dumps(gold)))
    return out


@pytest.mark.parametrize("name", sorted(_VENDORED))
def test_vendored_scorer_matches_canonical(name):
    path = _VENDORED[name]
    if not path.exists():
        pytest.skip(f"{name} baseline not present")
    vendored = _load(f"{name}_scoring", path)
    for pred, gold in _cases():
        assert vendored.score_precise(pred, gold)["primary"] == pytest.approx(
            _canonical(pred, gold)
        ), f"{name} scorer disagrees on pred={pred} gold={gold}"
