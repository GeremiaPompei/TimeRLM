# `anomalyxl` — the AnomalyXL generator

Synthetic long-context, multi-channel anomaly QA with **evidence-grounded** gold: every
answer is tied to a specific time span, channel, and magnitude, recovered exactly from
the synthesis metadata rather than from an LLM rating. This package produces the
AnomalyXL benchmark, and the same code is plugged into GRPO post-training as a task
generator.

## Design

A sample starts from a **scene**: `C` independent channels, each a base noise process
(white, red, or AR(1)) plus optional periodic and trend components, z-normalized to unit
std so an anomaly expressed in σ-units has a comparable scale everywhere. Channels are
anonymized (`ch_00`, `ch_01`, …) so no name priors leak.

An **anomaly** is one of five transforms — level shift, transient spike, change in
seasonality, change in variance, change in trend — applied to a contiguous window of one
channel *after* z-normalization.

The choice that makes long context hard: anomaly width is an **absolute** sample count
(8–256), independent of `L`. At `L = 131,072` an anomaly can occupy as little as 0.006%
of the series — indistinguishable from noise in a full-resolution plot unless you zoom.

## Tasks

Five **precise** families, each answered in strict JSON and scored by a continuous
metric in [0, 1]:

| Category | Answer | Metric |
|---|---|---|
| `localize` | `{"present": true, "start": …, "end": …}` | temporal IoU |
| `classify_with_evidence` | `{"kind": …, "start": …, "end": …}` | kind × IoU |
| `measure_magnitude` | `{"magnitude_sigma": …}` | relative error |
| `localize_all_channels` | `{"anomalies": [{"channel": …, "start": …, "end": …}, …]}` | set-matched F1 |
| `lead_lag_with_magnitude` | `{"direction": …, "lag_samples": …}` | direction + lag |

Each has a **coarse** multiple-choice projection, built by `scripts/coarse_project.py`
over the *identical* signal (`presence`, `start_time`, `categorization`, `magnitude`,
`identification`, `indicator`). Projecting rather than re-synthesizing is what lets the
paired design isolate the scoring axis with no different-signal confounder.

No-anomaly cases are first class: 20% of `localize` rows carry `present: false`, and
`localize_all_channels` draws `K ~ U{0..3}`, so ~25% of its rows have empty gold.

## Usage

```bash
uv sync

# The benchmark: 800 precise rows, then their 248-row coarse projection.
uv run python scripts/generate_anomalyxl_sweep.py --out-root datasets --precise-only --seed 42
uv run python scripts/coarse_project.py --src datasets/anomalyxl-precise --out datasets/anomalyxl-coarse

uv run pytest tests/ -q
```

`scripts/classical_baseline.py` is the no-LLM baseline (detector features + one
gradient-boosted head per question category). `scripts/export_anomalyxl_for_toto.py`
exports the data into the layout the Toto/ARFBench VLM baseline expects.
`generalization_set/` builds the real ECG and sleep transfer sets — see
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

## Difficulty knobs

Difficulty is configurable (`anomalyxl/difficulty.py`): magnitude range, width range,
channel count, context length, and the anomaly-kind mix. Two environment variables
override the negative rate, and RL post-training's first phase uses them to withhold
negatives from a cold-start policy. Both are read at **module import**, so they must be
exported in the launch environment:

- `ANOMALYXL_P_PRESENT` (default `0.8`) — probability a `localize` row has an anomaly.
- `ANOMALYXL_ALLCH_MIN_K` (default `0`) — minimum anomalies for `localize_all_channels`.
