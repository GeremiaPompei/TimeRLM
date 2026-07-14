"""Question templates registry.

Each `QuestionTemplate` subclass implements `build(length, n_channels, rng,
*, paired_split=None)` and is registered in `TEMPLATES` so the pipeline can
dispatch by `category`.

Two groups are registered. The five *precise* families are the open-ended
AnomalyXL tasks (strict JSON, continuous score); the six *coarse* templates are
the multiple-choice projections of those same five, built over the identical
signal by `scripts/coarse_project.py`.
"""

from __future__ import annotations

from .base import QuestionTemplate, Row
from .categorization import Categorization
from .classify_with_evidence import ClassifyWithEvidence
from .identification import Identification
from .indicator import Indicator
from .lead_lag_with_magnitude import LeadLagWithMagnitude
from .localize import Localize
from .localize_all_channels import LocalizeAllChannels
from .magnitude import Magnitude
from .measure_magnitude import MeasureMagnitude
from .presence import Presence
from .start_time import StartTime

TEMPLATES: dict[str, type[QuestionTemplate]] = {
    cls.category: cls
    for cls in (
        # Precise (open-ended, JSON answers).
        Localize,
        ClassifyWithEvidence,
        MeasureMagnitude,
        LocalizeAllChannels,
        LeadLagWithMagnitude,
        # Coarse (MCQ projections of the five above).
        Presence,
        StartTime,
        Categorization,
        Magnitude,
        Identification,
        Indicator,
    )
}

__all__ = ["QuestionTemplate", "Row", "TEMPLATES"]
