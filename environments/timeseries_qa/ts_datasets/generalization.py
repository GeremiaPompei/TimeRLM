"""Real-data generalization sets in the AnomalyXL schema.

These splice real LTAF (ECG) and Sleep-PSG (respiratory) recordings into
single-anomaly windows that match `anomalyxl-precise` exactly — every row is a
unified `classify_with_evidence` (positives carry the kind + window; negatives
carry kind "No anomaly" and no window), so they reuse AnomalyXL's parser, scorer,
and renderers verbatim. Each set is 250 windows (200 anomaly / 50 negative).

These classes only register dataset *names*; the parquet itself is either pulled
from the Hub or built from the raw PhysioNet recordings by
`anomalyXL/generalization_set/`.

Select them model-/mode-agnostically:

    vf-eval timeseries_qa -m <model> -a '{"dataset":"sleep_anomaly","mode":"rlm",
      "repo_or_path":"nz00shuuuu/anomalyxl-generalization#sleep-anomaly"}'

`repo_or_path` takes a local directory or a Hub shard:

    nz00shuuuu/anomalyxl-generalization#ltaf-anomaly
    nz00shuuuu/anomalyxl-generalization#sleep-anomaly

It can also be overridden with the per-dataset env var below.
"""

from __future__ import annotations

from .anomalyxl import AnomalyXL

_GEN_REPO = "nz00shuuuu/anomalyxl-generalization"


class LTAFAnomaly(AnomalyXL):
    """LTAF ECG: a single PVC/APC beat in NSR (classify) + clean-NSR negatives."""

    name = "ltaf_anomaly"
    default_repo_or_path = f"{_GEN_REPO}#ltaf-anomaly"
    env_var = "LTAF_ANOMALY_DATA_ROOT"


class SleepAnomaly(AnomalyXL):
    """Sleep-PSG: a single respiratory arousal (classify) + arousal-free negatives."""

    name = "sleep_anomaly"
    default_repo_or_path = f"{_GEN_REPO}#sleep-anomaly"
    env_var = "SLEEP_ANOMALY_DATA_ROOT"
