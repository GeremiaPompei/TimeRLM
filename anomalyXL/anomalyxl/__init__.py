"""`anomalyxl` — synthetic ARFBench-shaped multivariate anomaly QA.

Public surface re-exports the data types and pipeline entrypoints. See
`PLAN.md` for the design.
"""

from __future__ import annotations

from .anomalies import (
    ANOMALY_KINDS,
    COUPLING_KINDS,
    CouplingSpec,
    InjectionSpec,
    inject,
    make_coupling,
    make_injection,
)
from .pipeline import Row, SampleSpec, generate, specs_from_grid
from .questions import TEMPLATES, QuestionTemplate
from .scene import Scene, build_scene

__all__ = [
    "ANOMALY_KINDS",
    "COUPLING_KINDS",
    "CouplingSpec",
    "InjectionSpec",
    "QuestionTemplate",
    "Row",
    "SampleSpec",
    "Scene",
    "TEMPLATES",
    "build_scene",
    "generate",
    "inject",
    "make_coupling",
    "make_injection",
    "specs_from_grid",
]
