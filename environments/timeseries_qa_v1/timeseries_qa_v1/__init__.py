"""AnomalyXL RLM environment (verifiers v1).

A native v1 env for the numeric AnomalyXL RLM path: `AnomalyXLTaskset` (data + scoring,
reusing the v0 dataset) paired with `RLMLocalHarness` (runs the upstream `rlm` agent —
IPython kernel + auto-compaction — host-local). The framework resolves both from this
package's `__all__` by id, so the taskset id doubles as the harness id.
"""

from .generator import GeneratorCell, GeneratorConfig
from .harness import RLMLocalHarness, RLMLocalHarnessConfig
from .taskset import (
    AnomalyXLTask,
    AnomalyXLTaskset,
    AnomalyXLTasksetConfig,
    GeneratedAnomalyXLTask,
)

__all__ = [
    "AnomalyXLTask",
    "AnomalyXLTaskset",
    "AnomalyXLTasksetConfig",
    "GeneratedAnomalyXLTask",
    "GeneratorCell",
    "GeneratorConfig",
    "RLMLocalHarness",
    "RLMLocalHarnessConfig",
]
