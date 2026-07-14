"""Base class for a time-series QA dataset.

A subclass implements ``load_split`` and ``to_sample`` and declares its splits;
``build`` renders each sample into the columns the environment scores on. The
``render`` callable (text or image) is injected, so datasets don't depend on how
the series is presented.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Sequence

from datasets import Dataset

Series = Mapping[str, Sequence[float]]
Sample = dict  # {"series": Series, "question": str, "answer": str}
Renderer = Callable[[Sample], dict]  # sample -> input columns

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove complete ``<think>...</think>`` reasoning blocks before answer
    extraction. Reasoning models (e.g. Qwen3.5) emit their final answer *after*
    the think block; without stripping, an MCQ's bare final-line answer is never
    a candidate, and a precise JSON answer can be shadowed by a ``{...}`` written
    inside the reasoning. A no-op when no think block is present (e.g. GPT
    outputs), so it never perturbs scores for non-reasoning models."""
    if not text:
        return text
    return _THINK_RE.sub("", text)


class TimeSeriesQADataset(ABC):
    name: str
    splits: tuple[str, ...]
    default_split: str
    default_eval_split: str | None = None
    # Optional dataset-specific system prompt for text/image modes (None -> env default).
    system_prompt: str | None = None
    # RLM scaffolding variation points: extra sentence about value shape (e.g. NaN
    # gaps) and the exact answer-submission instruction. Defaults assume
    # parenthesised-letter golds like "(a)" / "(b)"; override on the subclass.
    rlm_task_preamble: str = (
        "This is a critical application: when relevant, precisely locate the anomaly's "
        "start and end using statistical thresholds and tools — be precise in your final answer. "
    )
    rlm_values_note: str = ""
    rlm_answer_format: str = "answer['content'] = '(x)'; answer['ready'] = True."
    # Extra dataset-specific guidance shown only when `enable_sub_llms=True`,
    # describing when `llm_batch` is worth reaching for on this data. Empty by
    # default — the env appends nothing.
    rlm_sub_llm_nudge: str = ""

    @abstractmethod
    def load_split(self, split: str) -> Dataset:
        """Raw rows for ``split`` as a HF ``Dataset``."""

    @abstractmethod
    def to_sample(self, row: dict) -> Sample:
        """Map a raw row to ``{"series", "question", "answer"}``. Datasets that
        ship pre-rendered images may also include an ``"image_uris"`` field
        (``list[str] | None``) consumed by the image renderer."""

    def build(self, split: str, render: Renderer) -> Dataset:
        raw = self.load_split(split)

        def to_columns(row: dict) -> dict:
            sample = self.to_sample(row)
            return {**render(sample), "answer": str(sample["answer"])}

        return raw.map(to_columns, remove_columns=raw.column_names)

    def score_answer(self, parsed_text: str, gold: str) -> float:
        """0/1 score for a parsed model output vs. the gold answer. Default is
        strict equality; subclasses override when a looser match is appropriate
        Gives freedom to each timeseries_qa dataset to score as it needs."""
        return 1.0 if parsed_text == gold else 0.0

    #: Names of additional per-row metrics the dataset reports, in fixed order.
    #: Envs (single-turn, RLM) wrap each name into a rubric func with weight 0
    #: — they're observation metrics, not reward contributions. Override to
    #: expose extras like ``acc_at_1pct`` or ``mae_samples``.
    extra_metric_names: tuple[str, ...] = ()

    def score_metrics(self, parsed_text: str, gold: str) -> dict[str, float]:
        """Compute the dataset's extra metrics for one (pred, gold) pair.

        Keys must be a subset of ``extra_metric_names``. The default returns
        empty — only datasets that declare extra metrics need to override.
        """
        return {}
