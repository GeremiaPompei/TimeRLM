"""Datadog/ARFBench — multiple-choice QA over real multivariate time series.

QA pairs (``arfbench-qa.csv``) ship separately from the series, which live as
``arfbench-ts-data/{query_group}_{interval}.parquet`` at six pre-downsampled
intervals. The finest interval whose serialized form fits the configured char
budget is picked per sample; the coarsest fallback is head/tail-truncated with
a visible marker. Downloads are lazy and cached by ``huggingface_hub``.

Series are pivoted to ``{"epoch": [...], "<channel>": [...]}`` so the model
sees per-tick timestamps — Anomaly Start / End golds are timestamps, so a
channel-only view cannot be answered correctly on those categories.

The system prompt body follows ARFBench's
``utils/prompt_test_benchmark.py:PROMPT_TEST_BENCHMARK``; the ``Input:`` line
is rewritten per render mode so the description matches what the model
actually receives. Score follows the spirit of
``utils/inference_utils.py:parse_response`` (JSON ``{"answer": ...}`` first,
then ``Answer:`` prefix, then strip-equality against gold), applied after
removing ``<think>`` blocks.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files

from .base import Sample, TimeSeriesQADataset, strip_reasoning

_REPO = "Datadog/ARFBench"
_INTERVALS = (10, 60, 300, 1800, 3600, 86400)  # seconds, finest first
_TRUNCATE_MARKER = "[...TRUNCATED FOR CONTEXT WINDOW...]"

# Kept as a standalone block: text/image modes ship it inside the system
# prompt, while RLM mode prepends it to the question itself — the RLM env
# builds its own system prompt (REPL instructions) and never sees
# ``ds.system_prompt``, yet Magnitude/Categorization questions are not
# answerable as specified without these definitions.
_CATEGORY_DEFINITIONS = """## Question Categories
There are 8 categories of questions:
1. Anomaly Presence

The anomaly presence question is a yes/no question that asks whether an anomaly is present in the time-series given.
An anomaly is present if the time-series has a value that is significantly different from the counterfactual values.

2. Anomaly Identification

The anomaly identification question asks the user to identify the channel of the anomaly in the time-series data, if an anomaly exists.
You must identify the correct channels referenced in the options, and decide based on the meaning of the time series description as well
as the context of the other channels to decide which channel(s) is anomalous.

3. Anomaly Start

The anomaly start question asks the user to identify the start time of the anomaly in the time-series data, if an anomaly exists.
The start time is the first time the anomaly appears in the time-series. If there is no exact timestamp for the start time,
the correct answer is the timestamp closest to the start of the anomaly.

4. Anomaly End

The anomaly end question asks the user to identify the end time of the anomaly in the time-series data, if an anomaly exists.
The end time is the last time the anomaly appears in the time-series. If there is no exact timestamp for the end time,
the correct answer is the timestamp closest to the end of the anomaly.

5. Anomaly Magnitude

The anomaly magnitude question asks the user to identify the magnitude of the anomaly in the time-series data, if an anomaly exists.
The magnitude is the maximum ratio of the anomaly values to the counterfactual non-anomalous values.
If the counterfactual values are 0, use the absolute deviation from the mean counterfactual values.
If there is no exact magnitude, the correct answer is the magnitude closest to the actual magnitude.

6. Anomaly Categorization

The anomaly categorization question asks the user to identify the category of the anomaly in the time-series data, if an anomaly exists.
There are 6 categories:
- Level Shift. This is when the time-series has a sustained change in mean value.
- Transient Spike. This is when the time-series has a sudden spike in value, but the value returns to the normal range after a very short period of time with no intervention.
- Change in Seasonality. This is when the time-series has a change in the seasonal pattern of the data.
- Change in Variance. This is when the time-series has a major sustained change in the variance of the data.
- Change in Trend. This is when the time-series has a major change in the trend (long term increase or decrease).
- No Anomaly

7. Anomaly Correlation

The anomaly correlation question is a paired query question that asks the user to identify whether the anomalies in two time-series
are correlated. Two anomalies are correlated if they have a known causal relation, if the time series have similar trends over time, or if they
have the same underlying root causes.

8. Anomaly Indicator

The anomaly indicator question is a paired query question that asks the user to identify whether some anomaly in the first time-series is
a leading or lagging indicator of the anomaly in the second time-series. Use the timing of the anomalies in the images to identify the correct answer.
"""

_ARFBENCH_PROMPT_BODY = """## Task Description
You are an expert at analyzing time-series data and answering questions about anomalies.
Your task is to answer the given question about time-series anomalies by selecting the most appropriate option.
Focus on the key aspects of the anomaly being analyzed and provide a clear explanation for your choice.

Input:
- Question: <question>
- Options: <options>
{input_section}

""" + _CATEGORY_DEFINITIONS.replace("{", "{{").replace("}", "}}") + """
## Answer Format
The answer should match one of the options exactly. Do not include the letter of the option in the answer. Include a detailed explanation of your reasoning for the answer.

The response MUST be a JSON in this format. Respond ONLY with the JSON. Do not include any extraneous formatting or Markdown quotes.
Output format:
{{
    "reasoning": <reasoning>,
    "answer": <answer>
}}
"""

_INPUT_SECTIONS = {
    "text": "- Time Series: The time series data, embedded per variate.",
    "image": "- PNG: PNG image of the time series.",
}


def _build_system_prompt(mode: str) -> str:
    return _ARFBENCH_PROMPT_BODY.format(
        input_section=_INPUT_SECTIONS.get(mode, _INPUT_SECTIONS["text"])
    )


class ARFBench(TimeSeriesQADataset):
    name = "arfbench"
    splits = ("train", "eval")
    default_split = "train"
    default_eval_split = "eval"
    # Gold is the full option string and the parquet contains ``null`` where
    # the source had NaN gaps — override the base defaults accordingly.
    # No anomaly-hunting preamble: the upstream ARFBench prompt is strictly
    # neutral ("... if an anomaly exists"), and the base default's "precisely
    # locate the anomaly's start and end" wording biases the model into always
    # finding one.
    rlm_task_preamble = ""
    rlm_values_note = "Values may be `null` where the source had NaN gaps. "
    rlm_answer_format = (
        "set answer['content'] to the chosen option exactly as it appears in the "
        "question (copy the text verbatim, no letter prefix), then set "
        "answer['ready'] = True."
    )

    def __init__(
        self,
        eval_size: int = 100,
        seed: int = 42,
        interval: int | None = None,
        max_context_chars: int = 100_000,
        shuffle_options: bool = True,
        mode: str = "text",
        difficulty: str | None = None,
        stratify: bool = False,
        indices: list[int] | None = None,
        data_dir: str | None = None,
    ):
        self.eval_size = eval_size
        self.seed = seed
        self.interval = interval  # pin a resolution; default None -> finest that fits
        self.max_context_chars = max_context_chars
        self.shuffle_options = shuffle_options
        self.mode = mode
        self.difficulty = difficulty  # e.g. "Tier 2": filter
        self.stratify = stratify
        self.indices = indices
        # Local checkout with the HF repo layout (arfbench-qa.csv,
        # arfbench-ts-data/, arfbench-images/). When set, no Hub request is
        # made — anonymous HF API calls get IP-rate-limited under heavy eval
        # traffic.
        self.data_dir = data_dir
        self.system_prompt = _build_system_prompt(mode)
        self._csv_path: str | None = None
        self._repo_files: set[str] | None = None
        self._series_cache: dict[tuple[str, int], dict[str, list]] = {}
        self._image_cache: dict[str, str] = {}

    def _resolve(self, name: str) -> str:
        """Path for a repo-relative file: local when ``data_dir`` is set,
        otherwise lazily downloaded from the Hub."""
        if self.data_dir:
            return os.path.join(self.data_dir, name)
        return hf_hub_download(_REPO, name, repo_type="dataset")

    def _qa_csv(self) -> str:
        if self._csv_path is None:
            self._csv_path = self._resolve("arfbench-qa.csv")
        return self._csv_path

    def _files(self) -> set[str]:
        if self._repo_files is None:
            if self.data_dir:
                root = Path(self.data_dir)
                self._repo_files = {
                    p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                }
            else:
                self._repo_files = set(list_repo_files(_REPO, repo_type="dataset"))
        return self._repo_files

    def load_split(self, split: str) -> Dataset:
        df = pd.read_csv(self._qa_csv())
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if self.difficulty:
            df = df[df["difficulty"] == self.difficulty]
            if df.empty:
                raise ValueError(
                    f"No rows with difficulty {self.difficulty!r}; "
                    "expected 'Tier 1', 'Tier 2', or 'Tier 3'."
                )
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        n_eval = min(self.eval_size, len(df))
        if self.stratify:
            in_eval = self._stratified_head_mask(df["task_category"], n_eval)
            df = df[in_eval] if split == "eval" else df[~in_eval]
        else:
            df = df.iloc[:n_eval] if split == "eval" else df.iloc[n_eval:]
        if self.indices is not None:
            df = df.iloc[self.indices]
        return Dataset.from_pandas(df, preserve_index=False)

    @staticmethod
    def _stratified_head_mask(categories: pd.Series, n: int) -> pd.Series:
        """Boolean mask picking ``n`` rows proportionally per category value
        (largest-remainder rounding), taking each category's earliest rows in
        the current — already shuffled — order. Deterministic for a given
        row order."""
        counts = categories.value_counts()
        quotas = counts * (n / len(categories))
        take = quotas.astype(int)
        for cat in (quotas - take).sort_values(ascending=False, kind="stable").index:
            if take.sum() >= n:
                break
            take[cat] += 1
        rank_within_category = categories.groupby(categories).cumcount()
        return rank_within_category < categories.map(take)

    def to_sample(self, row: dict) -> Sample:
        groups = [g.strip() for g in str(row["query_group"]).split(",") if g.strip()]
        options = self._parse_and_shuffle_options(row)
        options_str = json.dumps(options, ensure_ascii=False)
        question_text = f"{row['question']}\nOptions: {options_str}"
        series_budget = max(1, self.max_context_chars - len(question_text))
        series = self._select_series(groups, series_budget)
        if self.mode == "rlm":
            # Text/image modes carry the category definitions in the system
            # prompt; the RLM env never uses it, so ship them with the question
            # to keep the modes informationally equal.
            question_text = f"{_CATEGORY_DEFINITIONS}\n{question_text}"
        sample: Sample = {
            "series": series,
            "question": question_text,
            "answer": str(row["correct_answer"]),
        }
        # rlm mode also carries the official PNGs so the env can optionally
        # seed them into the first turn (hybrid image+REPL config).
        if self.mode in ("image", "rlm"):
            uris = self._compute_image_uris(groups, str(row.get("task_category") or ""))
            if uris:
                sample["image_uris"] = uris
        return sample

    def _compute_image_uris(self, groups: list[str], task: str) -> list[str]:
        # Mirrors upstream ``utils/inference_utils.py:get_image_paths``: combined
        # ``{g1}-{g2}.png`` for Indicator/Correlation, plus per-group ``{g}.png``.
        paths: list[str] = []
        files = self._files()
        if len(groups) == 2 and ("Indicator" in task or "Correlation" in task):
            combined = f"arfbench-images/{groups[0]}-{groups[1]}.png"
            if combined in files:
                paths.append(combined)
        for g in groups:
            name = f"arfbench-images/{g}.png"
            if name in files and name not in paths:
                paths.append(name)
        return [self._load_image_uri(p) for p in paths]

    def _load_image_uri(self, name: str) -> str:
        if name in self._image_cache:
            return self._image_cache[name]
        path = self._resolve(name)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        uri = f"data:image/png;base64,{data}"
        self._image_cache[name] = uri
        return uri

    def score_answer(self, parsed_text: str, gold: str) -> float:
        gold = gold.strip()
        for candidate in self._answer_candidates(strip_reasoning(parsed_text or "")):
            if candidate.strip() == gold:
                return 1.0
        return 0.0

    @staticmethod
    def _answer_candidates(text: str):
        yield text
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            try:
                obj = json.loads(text[start:end])
            except (json.JSONDecodeError, TypeError, ValueError):
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                yield obj["answer"]
        m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
        if m:
            yield m.group(1).strip().strip("\"'")

    def _parse_and_shuffle_options(self, row: dict) -> list[str]:
        raw = row.get("options_str") or row.get("options")
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        options: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                options.append(item)
            elif isinstance(item, dict) and "value" in item:
                options.append(str(item["value"]))
        if self.shuffle_options and options:
            # Per-row deterministic shuffle: stable across runs, no global state.
            rng = random.Random(f"{self.seed}:{row.get('question', '')}")
            rng.shuffle(options)
        return options

    def _select_series(self, groups: list[str], budget: int) -> dict[str, list]:
        candidates = (self.interval,) if self.interval else _INTERVALS
        # Image / RLM modes pay no chat-prompt cost for series size — data goes
        # out-of-band (PNG or sandbox file). Skip the budget ladder and the
        # truncation marker: numpy on a list with a sentinel string would
        # break, and matplotlib doesn't care about point count.
        if self.mode != "text":
            for interval in candidates:
                try:
                    return self._load_groups(groups, interval)
                except FileNotFoundError:
                    continue
            raise FileNotFoundError(f"No parquet for query groups {groups!r}.")
        coarsest: dict[str, list] | None = None
        for interval in candidates:
            try:
                series = self._load_groups(groups, interval)
            except FileNotFoundError:
                continue
            coarsest = series
            if self._estimate_series_chars(series) <= budget:
                return series
        if coarsest is None:
            raise FileNotFoundError(f"No parquet for query groups {groups!r}.")
        return self._truncate_to_budget(coarsest, budget)

    def _load_groups(self, groups: list[str], interval: int) -> dict[str, list]:
        out: dict[str, list] = {}
        multi = len(groups) > 1
        for qg in groups:
            for channel, values in self._load_group_at(qg, interval).items():
                key = f"{qg} | {channel}" if multi else channel
                out[key] = values
        return out

    def _load_group_at(self, query_group: str, interval: int) -> dict[str, list]:
        cache_key = (query_group, interval)
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]
        name = f"arfbench-ts-data/{query_group}_{interval}.parquet"
        if name not in self._files():
            raise FileNotFoundError(name)
        path = self._resolve(name)
        df = pd.read_parquet(path).sort_values("epoch")
        # All channels in one parquet share the same epoch grid. Lead with
        # ``epoch`` so the model can answer timestamp-valued categories
        # (Anomaly Start / End).
        epochs = df["epoch"].drop_duplicates().sort_values().astype(str).tolist()
        pivoted: dict[str, list] = {"epoch": epochs}
        for group, sub in df.groupby("group"):
            # Univariate series carry an empty group tag; a blank channel key
            # would confuse the RLM ("read the actual channel keys") and text
            # renders alike.
            pivoted[str(group) or "value"] = [v if v == v else None for v in sub["value"].tolist()]
        self._series_cache[cache_key] = pivoted
        return pivoted

    @staticmethod
    def _estimate_series_chars(series: dict[str, list]) -> int:
        # Approximation of format_timeseries() output: "name: v1, v2, ...\n".
        total = 0
        for name, values in series.items():
            total += len(name) + 2
            for v in values:
                total += len(str(v)) + 2
            total += 1
        return total

    @staticmethod
    def _truncate_to_budget(series: dict[str, list], budget: int) -> dict[str, list]:
        n = max(1, len(series))
        per_channel = max(200, budget // n)
        out: dict[str, list] = {}
        for name, values in series.items():
            values = list(values)
            probe = values[: min(32, len(values))] or [0]
            avg_chars = max(2, sum(len(str(v)) + 2 for v in probe) // len(probe))
            target = max(20, (per_channel - len(name) - 4) // avg_chars)
            if target >= len(values):
                out[name] = values
                continue
            head = int(target * 0.85)
            tail = target - head
            kept: list = values[:head] + [_TRUNCATE_MARKER]
            if tail:
                kept.extend(values[-tail:])
            out[name] = kept
        return out
