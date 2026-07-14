# timeseries-qa

The evaluation environment. Each sample is a time series (one or more channels) plus a
question and an evidence-grounded gold answer. The same dataset is presented to the model
in one of three ways, which is what makes the paper's four inference-time configurations
a controlled comparison:

- **`text`** — the series is serialized into the prompt (single-pass). Long series are
  uniformly decimated to fit the context window.
- **`image`** — the series is rendered as a plot (single-pass).
- **`rlm`** — **TimeRLM**. The series is handed to the model as a file in a sandboxed
  Python REPL (`numpy` + `scipy`), which it queries over multiple turns at full
  resolution. With `use_image=true` a `look(path)` tool is registered: the model saves a
  plot in a REPL call, then calls `look` to inject the image inline into its next turn.
  This is the *only* code path that produces the image RLM variant.

Scoring is **not** 0/1 exact match. Each category is scored by its own continuous metric
in [0, 1] — temporal IoU, kind × IoU, relative error, set-matched F1, or a
direction-and-lag composite — and a response that is not valid structured output scores
zero, since producing machine-checkable evidence is itself part of the task. The same
scorers are used at training time as the RL reward, so training and evaluation cannot
drift apart.

### Datasets

Selected with the `dataset` arg; each is a `TimeSeriesQADataset` subclass under
`ts_datasets/`. The released shards live in the
[TimeRLM / AnomalyXL collection](https://huggingface.co/collections/nz00shuuuu/timerlm-model-and-anomalyxl-datasets).

| `dataset` | What it is | `repo_or_path` |
| --- | --- | --- |
| `anomalyxl` | AnomalyXL **precise**: the five open-ended, evidence-grounded task families | `nz00shuuuu/AnomalyXL#Localize/test` |
| `anomalyxl_coarse` | AnomalyXL **coarse**: the MCQ projection of the same signals | `nz00shuuuu/AnomalyXL#MCQ/test` |
| `ltaf_anomaly` | Real ECG (PhysioNet LTAF), 250 windows, in the AnomalyXL schema | `nz00shuuuu/anomalyxl-generalization#ltaf-anomaly` |
| `sleep_anomaly` | Real sleep (Sleep-PSG), 250 windows, in the AnomalyXL schema | `nz00shuuuu/anomalyxl-generalization#sleep-anomaly` |
| `arfbench` | [Datadog/ARFBench](https://huggingface.co/datasets/Datadog/ARFBench), real observability telemetry. Pass `difficulty="Tier 2"` for the 306 MCQ questions the paper reports | (fetched by the loader) |

`repo_or_path` accepts either a **local directory** holding `data.parquet` (+ optional
`manifest.json`), or an **HF dataset repo id** with an optional `#fragment` selecting what
to pull: a partition/split parquet (`#Localize/test`, `#MCQ/val` in the AnomalyXL
monorepo), a subfolder (`#ltaf-anomaly`), or a bare split (`#test`). A local path always
wins over a repo id. The AnomalyXL shards are also regenerable from scratch — see
[`../../anomalyXL/`](../../anomalyXL/).

### Key arguments

Passed as JSON via `-a` / `--env-args`.

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset` | str | `"anomalyxl"` | One of the datasets above |
| `mode` | str | `"text"` | `text`, `image`, or `rlm` |
| `use_image` | bool | `true` | **RLM mode only.** `true` registers the `look()` image tool; `false` is a pure-numeric REPL |
| `use_local_executor` | bool | `false` | Run the REPL host-locally instead of in a remote sandbox |
| `max_turns` | int | `null` | Cap on agent turns (the paper uses 15 during RL and its evals) |
| `repo_or_path` | str | required | Local directory with `data.parquet`, or an HF dataset repo id with an optional `#fragment` (see above) |
| `eval_size` | int | dataset default | Size of the eval pool: the **first** `eval_size` rows (`df.iloc[:eval_size]`, clamped to the row count); the remainder becomes the train pool. Pass the full row count to evaluate the whole shard (800 precise / 248 coarse / 250 generalization). `eval_size: 0` evaluates **nothing** |
| `enable_sub_llms` | bool | `false` | Gives the root model an `llm_batch()` tool for depth-1 sub-RLM calls. Implemented but **disabled throughout the paper** |
| `seed` | int | `42` | Seed |

### Quickstart

```bash
uv sync

# TimeRLM, text variant, on the AnomalyXL precise shard.
.venv/bin/vf-eval timeseries_qa -m <model> -b http://localhost:8000/v1 -n 800 -c 12 -s \
  -a '{"dataset":"anomalyxl","repo_or_path":"../../anomalyXL/datasets/anomalyxl-precise","mode":"rlm","use_image":false,"use_local_executor":true,"eval_size":800}'
```

Each run writes `metadata.json` with `avg_reward` and the per-category `avg_metrics`.
See [`../../docs/REPRODUCE.md`](../../docs/REPRODUCE.md) for all four configurations.

### A note on `rlm_env_vendored.py`

`rlm.py` builds on `RLMEnv`, which upstream `verifiers` removed in PR #1515.
`rlm_env_vendored.py` is a frozen copy of that module from the last commit before its
removal, and `rlm.py` / `local_executor.py` fall back to it when the upstream one is
absent. It is load-bearing for both eval and RL — do not delete it.
