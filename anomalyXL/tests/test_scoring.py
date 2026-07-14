from __future__ import annotations

import pytest

from anomalyxl.scoring import semantic_class


def test_categorization_binning():
    assert semantic_class("categorization", "Level Shift") == "Level Shift"
    assert semantic_class("categorization", "Change in Variance") == "Change in Seasonality or Variance"
    assert semantic_class("categorization", "Change in Seasonality") == "Change in Seasonality or Variance"
    assert semantic_class("categorization", "No Anomaly") == "No Anomaly"


def test_presence_binning():
    assert semantic_class("presence", "Yes") == "Yes"
    assert semantic_class("presence", "No") == "No"


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        semantic_class("nope", "anything")
