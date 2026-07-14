"""Shared types for question templates.

`Row` is the in-memory output of `QuestionTemplate.build`; `pipeline.py`
flattens it into the parquet schema in `PLAN.md §7`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..anomalies import CouplingSpec, InjectionSpec

__all__ = ["QuestionTemplate", "Row"]


@dataclass(frozen=True)
class Row:
    series: dict[str, list[float]]
    question: str
    options: list[str]
    answer: str
    semantic_class: str
    injections: list[InjectionSpec] = field(default_factory=list)
    coupling: CouplingSpec | None = None


class QuestionTemplate(ABC):
    category: str
    tier: int
    # Minimum channel count this template can be built at. For paired (Tier III)
    # templates this is the *total* channel budget (A + B). `specs_from_grid`
    # filters out incompatible (category, n_channels) cells using this.
    min_channels: int = 1

    @abstractmethod
    def build(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        paired_split: tuple[float, float] | None = None,
    ) -> Row:
        """Synthesize a scene + question + gold answer for this category.

        Templates own scene construction so they can condition synthesis on
        the gold (e.g. Presence "no" forbids any injection; Categorization
        picks the kind first and conditions injection on it). Tier III
        templates split `n_channels` into `(A, B)` per `paired_split`.
        """

    @abstractmethod
    def semantic_class(self, gold: str) -> str:
        """Map the gold option text to its semantic-bin label for macro-F1."""
