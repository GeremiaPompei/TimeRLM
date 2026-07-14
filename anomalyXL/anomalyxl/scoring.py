"""Semantic-bin mapping for macro-F1 — mirrors ARFBench Appendix C.2.

The actual bin definitions live on each `QuestionTemplate.semantic_class`;
this module is a thin dispatcher so analysis code can reduce a `(category,
gold)` pair to a bin label without instantiating the template tree.
"""

from __future__ import annotations

from .questions import TEMPLATES

__all__ = ["semantic_class"]


def semantic_class(category: str, gold: str) -> str:
    """Bin label for a `(category, gold-option-text)` pair.

    The category's template owns the bin definition; we just look it up.
    """
    if category not in TEMPLATES:
        raise KeyError(f"Unknown category {category!r}; available: {sorted(TEMPLATES)}")
    return TEMPLATES[category]().semantic_class(gold)
