# TimeRLM: Recursive Language Models Are Precise Temporal Reasoners

Code and benchmark for **TimeRLM**, a Recursive Language Model formulation for
time-series, and **AnomalyXL**, a synthetic long-context anomaly-localization benchmark
released together with its generator.

🤗 **[TimeRLM / AnomalyXL datasets on Hugging Face](https://huggingface.co/collections/nz00shuuuu/timerlm-model-and-anomalyxl-datasets)**

TimeRLM treats a long recording as an environment variable rather than something to
ingest: the signal lives in a file in a sandboxed Python REPL, and the model queries it
over multiple turns with `numpy`/`scipy` — and, in the image variant, with a `look()`
tool that renders plots back into context — before committing a structured,
evidence-grounded answer. This lets it operate on the full, un-decimated series instead
of a single compressed or subsampled view.

![The TimeRLM loop](public/RLM_Loop.png)

*The orchestrator reasons in natural language and emits Python into a sandboxed REPL,
where the signal is an environment variable. `numpy` and `scipy` are common to all
variants; including `matplotlib` exposes a `look()` tool that inserts plots into the
model's context. A `FINAL()` action commits a grounded answer.
([vector version](public/RLM_Loop.pdf))*

## What's here

| Directory | What it is |
|---|---|
| `anomalyXL/` | The **AnomalyXL generator**. Five precise task families (localize, classify-with-evidence, measure-magnitude, localize-all-channels, lead–lag), each with a coarse MCQ projection over the *same* signal. Also the ECG/sleep generalization-set builders and the classical no-LLM baseline. |
| `environments/timeseries_qa/` | The **evaluation environment** (`verifiers`, id `timeseries-qa`). One env, four inference-time configurations: single-pass text, single-pass image, TimeRLM text, TimeRLM image. Also the ARFBench and generalization-set loaders. |
| `environments/timeseries_qa_v1/` | The **RL training environment** (id `timeseries-qa-v1`): the generator-backed taskset, the reward (the benchmark's own scorer, plus the −0.25 turn-budget penalty), and the harness that drives the agent. |
| `rlm-harness/` | The **agent loop**: a persistent IPython sandbox with auto-compaction and a turn budget. Vendored from Prime Intellect's `rlm` — see `rlm-harness/NOTICE`. |
| `configs/rl/` | The **GRPO post-training config** behind the paper's RL row. |
| `sft/tslm/` | The **TSLM baselines**: standalone SFT + eval for ChatTS, OpenTSLM-Flamingo and ITFormer, all on a frozen Chronos-2 encoder feeding a LoRA-tuned Qwen3.5-4B. |
| `scripts/` | Bootstrap, data splitting, checkpoint eval, and the generalization-set scorer. |

Two RLM code paths exist by design and are both load-bearing: evaluation runs through
`environments/timeseries_qa` (which owns the `look()` vision tool), while RL training
runs through `environments/timeseries_qa_v1` + `rlm-harness`.

## Quickstart

```bash
bash scripts/setup.sh                  # prime-rl + the three packages
cd anomalyXL && uv sync
uv run python scripts/generate_anomalyxl_sweep.py --out-root datasets --precise-only --seed 42
uv run pytest tests/ -q                # 157 tests, no GPU needed
```

The full walkthrough — building the benchmark, building the real ECG/sleep
generalization set, running all four inference-time configurations, and reproducing the
GRPO run — is in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

## The benchmark

![AnomalyXL: evidence-grounded anomaly localization in long context](public/AnomalyXL_zoom.png)

*Five task families — localize, classify, measure magnitude, localize across all
channels, and lead–lag — each posed in two paired formulations over the **same** signal: a
`precise` open-ended version scored on class and IoU, and a `coarse` multiple-choice
projection. ([vector version](public/AnomalyXL_zoom.pdf))*

AnomalyXL is synthesized programmatically, so every gold label is exact and strict
scoring is well-defined. Difficulty is configurable along context length (16k–131k
samples), channel count (1–16), and anomaly kind, and 20% of samples are first-class
negatives. Anomaly width is an *absolute* sample count (8–256) regardless of series
length — which is what makes long context hard: at L = 131,072 an anomaly can occupy as
little as 0.006% of the series.

Every item is posed twice over the identical signal: a **precise**, open-ended version
scored by a continuous metric (temporal IoU, kind×IoU, relative error, set-matched F1,
or a direction-and-lag composite), and a **MCQ** multiple-choice projection. Holding
the signal fixed isolates the effect of evidence-grounded scoring from that of the
multiple-choice format, with no different-signal confounder.

The released shards — the `AnomalyXL` monorepo (`Localize` precise + `MCQ` coarse
partitions) and the real ECG/sleep `anomalyxl-generalization` set — are collected on
🤗 [Hugging Face](https://huggingface.co/collections/nz00shuuuu/timerlm-model-and-anomalyxl-datasets).

## What's not included

To be explicit about the boundary of this artifact:

- **The prime-rl patches** used for the paper's RL run. `scripts/setup.sh` clones upstream
  prime-rl; the exact commit and patch set are not pinned here (see `docs/REPRODUCE.md`).
- **The PhysioNet preprocessing** that produces the derived layout the generalization-set
  builders consume. Use the released Hub shards instead.
- **Model weights.** The fine-tuned LoRA adapters are not committed; `sft/toto/eval.py`
  and the TSLM configs train them from scratch.

Every baseline row in the paper *is* reproducible from this repo: the three TSLMs
(`sft/tslm/`), Toto (`sft/toto/`), and the classical no-LLM detector
(`anomalyXL/scripts/classical_baseline.py`). All four are scored by the same scorer as
TimeRLM — enforced by `environments/timeseries_qa/tests/test_scorer_parity.py`, which
fails if the subprojects' vendored copies ever drift apart.

## Acknowledgements

- **`rlm`** ([PrimeIntellect-ai/rlm](https://github.com/PrimeIntellect-ai/rlm)) — the
  recursive agent harness, vendored here as `rlm-harness/` (see its `NOTICE`), together
  with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and
  [verifiers](https://github.com/PrimeIntellect-ai/verifiers).
- **PhysioNet** — the Long Term AF Database and the 2018 Sleep-PSG challenge set, from
  which the generalization windows are spliced.

## License

MIT — see [LICENSE](LICENSE). `rlm-harness/` is a vendored copy of Prime Intellect's `rlm`
(MIT, see `rlm-harness/NOTICE`); `rlm_env_vendored.py` is frozen from `verifiers` (MIT);
ARFBench and the PhysioNet recordings retain their own licenses.
