"""Env-side tests for `ts_datasets/arfbench.py`.

Coverage (offline — series loading is stubbed where needed):
- `score_answer`: exact match, JSON ``{"answer": ...}``, ``Answer:`` prefix,
  and <think>-block stripping (a brace inside the reasoning must not shadow
  the answer JSON emitted after it).
- Prompt parity: text/image system prompts carry the category definitions;
  RLM mode prepends the same block to the question instead (the RLM env
  never delivers ``ds.system_prompt``).
"""

from __future__ import annotations

import json

from ts_datasets.arfbench import _CATEGORY_DEFINITIONS, ARFBench


def _ds(**kwargs) -> ARFBench:
    return ARFBench(**kwargs)


# --------------------------------------------------------------- score_answer

GOLD = "Yes, there is an anomaly in the time-series"


def test_score_exact_match():
    assert _ds().score_answer(GOLD, GOLD) == 1.0


def test_score_json_answer():
    pred = json.dumps({"reasoning": "spike at t=5", "answer": GOLD})
    assert _ds().score_answer(pred, GOLD) == 1.0


def test_score_answer_prefix():
    assert _ds().score_answer(f"Answer: {GOLD}", GOLD) == 1.0


def test_score_wrong_answer():
    assert _ds().score_answer("No, there is no anomaly", GOLD) == 0.0


def test_score_think_block_with_braces_does_not_shadow_answer():
    """A `{` inside <think> must not anchor JSON extraction at the wrong spot."""
    pred = (
        "<think>maybe {check the spike} first</think>\n"
        + json.dumps({"reasoning": "x", "answer": GOLD})
    )
    assert _ds().score_answer(pred, GOLD) == 1.0


def test_score_empty_and_none_safe():
    assert _ds().score_answer("", GOLD) == 0.0
    assert _ds().score_answer(None, GOLD) == 0.0


# --------------------------------------------------------------- prompt parity

ROW = {
    "query_group": "g1",
    "question": "Does the time-series exhibit an anomaly?",
    "options_str": json.dumps([GOLD, "No anomaly"]),
    "correct_answer": GOLD,
}


def _stub_series(ds: ARFBench) -> None:
    ds._select_series = lambda groups, budget: {"epoch": ["0", "1"], "value": [1.0, 2.0]}


def test_text_image_system_prompts_carry_definitions():
    for mode in ("text", "image"):
        prompt = _ds(mode=mode).system_prompt
        assert "## Question Categories" in prompt
        assert "maximum ratio of the anomaly values" in prompt  # magnitude rule


def test_rlm_question_carries_definitions():
    ds = _ds(mode="rlm")
    _stub_series(ds)
    sample = ds.to_sample(dict(ROW))
    assert sample["question"].startswith(_CATEGORY_DEFINITIONS)
    assert ROW["question"] in sample["question"]


def test_text_question_has_no_duplicate_definitions():
    ds = _ds(mode="text")
    _stub_series(ds)
    sample = ds.to_sample(dict(ROW))
    assert "## Question Categories" not in sample["question"]


# --------------------------------------------------------- stratified sampling

def test_stratified_mask_proportional_counts():
    import pandas as pd

    cats = pd.Series(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    mask = ARFBench._stratified_head_mask(cats, 10)
    assert mask.sum() == 10
    picked = cats[mask].value_counts()
    assert picked["a"] == 6 and picked["b"] == 3 and picked["c"] == 1


def test_stratified_mask_largest_remainder_exact_n():
    import pandas as pd

    # Quotas 3.33 / 3.33 / 3.33 -> floors sum to 9, one remainder seat.
    cats = pd.Series(["a", "b", "c"] * 30)
    mask = ARFBench._stratified_head_mask(cats, 10)
    assert mask.sum() == 10
    assert set(cats[mask].value_counts()) == {4, 3}


def test_indices_select_positions_within_split(monkeypatch):
    import pandas as pd

    qa = pd.DataFrame({
        "question": [f"q{i}" for i in range(10)],
        "task_category": ["Anomaly Presence"] * 10,
        "difficulty": ["Tier 1"] * 10,
        "options_str": ["[]"] * 10,
        "correct_answer": ["x"] * 10,
        "query_group": ["g"] * 10,
    })
    base = ARFBench(eval_size=5)
    full = None

    def fake_csv(self):
        nonlocal full
        return "unused"

    monkeypatch.setattr(ARFBench, "_qa_csv", fake_csv)
    monkeypatch.setattr(pd, "read_csv", lambda _: qa.copy())

    full_eval = base.load_split("eval").to_pandas()
    picked = ARFBench(eval_size=5, indices=[1, 3]).load_split("eval").to_pandas()
    assert list(picked["question"]) == [full_eval["question"][1], full_eval["question"][3]]


def test_stratified_mask_takes_earliest_rows_per_category():
    import pandas as pd

    cats = pd.Series(["a", "a", "b", "a", "b", "b"])
    mask = ARFBench._stratified_head_mask(cats, 2)
    # One per category, earliest occurrence of each.
    assert list(mask[mask].index) == [0, 2]
