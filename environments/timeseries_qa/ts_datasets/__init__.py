"""Registry of available time-series QA datasets.

To add a dataset: create a module in this package with a ``TimeSeriesQADataset``
subclass, then list its class below. ``DATASETS`` maps name -> class.
"""

from .anomalyxl import AnomalyXL
from .arfbench import ARFBench
from .base import TimeSeriesQADataset
from .generalization import LTAFAnomaly, SleepAnomaly
from .anomalyxl_coarse import AnomalyXLCoarse

_CLASSES: tuple[type[TimeSeriesQADataset], ...] = (
    ARFBench, AnomalyXLCoarse, AnomalyXL, LTAFAnomaly, SleepAnomaly,
)

DATASETS: dict[str, type[TimeSeriesQADataset]] = {cls.name: cls for cls in _CLASSES}

__all__ = ["DATASETS", "TimeSeriesQADataset"]
