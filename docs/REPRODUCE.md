# Reproducing TimeRLM

Everything the paper reports comes from four artifacts in this repo: the AnomalyXL
generator, the real-signal generalization set, the TimeRLM agent harness, and the GRPO
post-training config. This document walks each one end to end.

Paths below are relative to the repo root unless stated otherwise. RL commands run from
`./prime-rl` (that is why the config's `repo_or_path` values start with `../`).

## 0. Machine and bootstrap

A Linux GPU box with a recent NVIDIA driver (≥ 570 for the CUDA 12.8 wheels prime-rl
pins). `uv` handles Python and every venv.

```bash
sudo apt update && sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <this-repo> TimeRLM && cd TimeRLM
bash scripts/setup.sh
```

`scripts/setup.sh` clones prime-rl into `./prime-rl` (gitignored), syncs its venv, and
installs the three packages this repo ships — `anomalyxl` (the generator),
`timeseries_qa` (the eval env, id `timeseries-qa`), and `timeseries_qa_v1` (the RL
taskset + harness, id `timeseries-qa-v1`) — into that venv.

## 1. Get the data

Two options: **download the released shards**, or **regenerate them** from the generator
in this repo. Both give the same data; the generator is the source of truth.

### Option A — download from the Hub

Both datasets are public, and collected here:
**[TimeRLM / AnomalyXL datasets](https://huggingface.co/collections/nz00shuuuu/timerlm-model-and-anomalyxl-datasets)**

| Dataset | Contents |
|---|---|
| [`nz00shuuuu/AnomalyXL`](https://huggingface.co/datasets/nz00shuuuu/AnomalyXL) | The synthetic benchmark, one monorepo with two partitions: `Localize` (the five open-ended precise task families) and `MCQ` (the coarse projection of the same signals). Each ships `train` / `val` / `test` (the test split is the benchmark). |
| [`nz00shuuuu/anomalyxl-generalization`](https://huggingface.co/datasets/nz00shuuuu/anomalyxl-generalization) | The real-signal transfer sets: `ltaf-anomaly` and `sleep-anomaly`, 250 windows each. |

You do not need to download these by hand. Anywhere the env takes a `repo_or_path`, you
can pass a Hub shard directly, using a `#fragment` to pick what you want out of the repo:

```
nz00shuuuu/AnomalyXL#Localize/test              # the precise benchmark test split
nz00shuuuu/AnomalyXL#Localize/val               # the online-eval split
nz00shuuuu/AnomalyXL#MCQ/test                   # the coarse regime
nz00shuuuu/anomalyxl-generalization#ltaf-anomaly
nz00shuuuu/anomalyxl-generalization#sleep-anomaly
```

A local directory always wins over a repo id, so an existing local checkout is never
shadowed by a download. Without a fragment, split-style repos default to `test`.

### Option B — regenerate from the generator

The generator is the benchmark: nothing is downloaded, and every gold label is exact.

```bash
cd anomalyXL
uv sync

# Precise shard (800 rows, seed 42): the 5 open-ended task families.
uv run python scripts/generate_anomalyxl_sweep.py --out-root datasets --precise-only --seed 42

# Coarse shard (248 rows): the MCQ projection of the SAME signals. This is what makes
# the coarse-vs-precise comparison free of a different-signal confounder.
uv run python scripts/coarse_project.py --src datasets/anomalyxl-precise --out datasets/anomalyxl-coarse

# Train/val splits, for the SFT'd baselines and the RL run's online eval.
cd .. && uv run python scripts/split_anomalyxl_train_val.py \
    --src anomalyXL/datasets/anomalyxl-precise
```

Defaults reproduce the paper's composition: context lengths 16k–131k, 1–16 channels,
five anomaly kinds, and **20% hard negatives** (`localize` draws `present=False` at
`1 - ANOMALYXL_P_PRESENT = 0.2`; `localize_all_channels` draws `K ~ U{0..3}`, so ~25% of
rows carry empty gold).

## 2. Rebuild the generalization set (optional — real ECG + sleep)

Real recordings, spliced into the AnomalyXL schema and scored by the identical parser
and metric. No synthetic signal appears here, so it measures zero-shot transfer.

The released shards are on the Hub (`nz00shuuuu/anomalyxl-generalization#ltaf-anomaly`
and `#sleep-anomaly`), so this step is only needed if you want to rebuild them from the
raw recordings. To do that, download the two PhysioNet sources yourself — the Long Term
AF Database and the 2018 Sleep-PSG challenge set — and point the builders at them.

Each set is built in two stages: the short windows (60–300 s) first, then the
long-context ones (≥ 300 s), drawn under strict per-kind quotas and merged into the final
250-row set the environment loads.

```bash
cd anomalyXL/generalization_set
export LTAF_DATA_ROOT=/path/to/ltafdb SLEEP_DATA_ROOT=/path/to/sleep_psg

# Stage 1 — short windows -> datasets/{ltaf,sleep}-anomaly-short/
uv run python gen_ltaf.py  --n 100
uv run python gen_sleep.py --n 100

# Stage 2 — long windows, merged -> datasets/{ltaf,sleep}-anomaly/  (250 rows each)
# Without --write it is a dry run that only reports whether the quotas can be met.
uv run python extend_ltaf.py  --write
uv run python extend_sleep.py --write
```

Both stages also accept an explicit `--data-root` instead of the env vars.

## 3. Evaluate

All four inference-time configurations run through one env, `timeseries-qa`, switched by
`mode` (`text` | `image` | `rlm`) and `use_image`. Single-pass text decimates the series
into the prompt; single-pass image renders it as a plot; both RLM variants reach the
full-resolution signal through the REPL.

Serve the model (or point `-b` at any OpenAI-compatible endpoint), then:

```bash
cd environments/timeseries_qa && uv sync
PRECISE=nz00shuuuu/AnomalyXL#Localize/test       # or a local directory

# TimeRLM, text variant — the paper's headline configuration.
.venv/bin/vf-eval timeseries_qa -m <model> -b http://localhost:8000/v1 -n 800 -c 12 -s \
  -o outputs/precise-rlm-text \
  -a "{\"dataset\":\"anomalyxl\",\"repo_or_path\":\"$PRECISE\",\"mode\":\"rlm\",\"use_image\":false,\"use_local_executor\":true,\"eval_size\":800}"

# Single-pass text:  mode=text
# Single-pass image: mode=image        (use_image does NOT apply outside rlm mode)
# TimeRLM image:     mode=rlm, use_image=true   (registers the look() tool)
# Coarse regime: dataset "anomalyxl_coarse", repo_or_path nz00shuuuu/AnomalyXL#MCQ/test.
```

`eval_size` is the size of the eval pool, taken as the FIRST `eval_size` rows
(`df.iloc[:eval_size]`, clamped to the row count); the remainder becomes the train pool.
Pass the full row count to evaluate the whole shard — `eval_size: 800` above, or 248 for
the coarse shard and 250 for a generalization shard. `eval_size: 0` evaluates NOTHING.

`metadata.json` in each output directory carries the per-category `avg_metrics` that
populate Table 1. Read the `correct_answer` metric, not `avg_reward`: in RLM mode
`avg_reward` also includes a `length_efficiency` term at weight 0.1 (`rlm.py`), which the
single-pass rubric does not have — so `avg_reward` is not comparable across
configurations.

**Generalization set:** same command with `"dataset":"ltaf_anomaly"` or
`"sleep_anomaly"`, and `repo_or_path` set to
`nz00shuuuu/anomalyxl-generalization#ltaf-anomaly` (or `#sleep-anomaly`). Score the
runs with `scripts/report_generalization.py`, which emits the classification-F1 / IoU /
Total breakdown.

**ARFBench Tier 2:** `"dataset":"arfbench"`, `"difficulty":"Tier 2"`, `"eval_size":306`.
Pass `eval_size` explicitly — it defaults to 100, which would score only part of the tier.
We reproduce ARFBench's Tier-2 prompt: the task description, the category definitions, the
question, and the option set are their text, and the answer must match an option exactly.
Two things differ from their release, both deliberate: options are **shuffled**
(`"shuffle_options": false` to disable), and the answer-format instruction is adapted per
harness — in RLM mode the model commits its answer through the REPL rather than by
emitting their `{"reasoning", "answer"}` JSON, so the category definitions ride in the
question instead of the system prompt, keeping the modes informationally equal. The
benchmark itself is [Datadog/ARFBench](https://github.com/Datadog/ARFBench).

**TSLM baselines (ChatTS, OpenTSLM-Flamingo, ITFormer):** `sft/tslm/`. All three share a
frozen Chronos-2 encoder feeding a LoRA-tuned `Qwen/Qwen3.5-4B`, and each is fine-tuned
separately for the coarse and the precise regime — six runs in total.

```bash
cd sft/tslm && uv sync
uv run python fetch_data.py                                # train/val/test shards from the Hub
uv run python train.py --config configs/chatts_precise.yaml    # x6 configs
uv run python aggregate_results.py --results-dir results --out anomalyxl_tslm.tsv
```

They train on the 5,008-row precise shard (4488 train + 520 val) and are evaluated on the
same 800-row gold test shard, parsed and scored by the **same scorer** as every other row
(`sft/tslm/src/datasets/anomaly_xl/scoring.py` is a vendored copy of the environment's
scorer, verified to agree with it on all five categories).

**Toto (TSFM–VLM hybrid):** `sft/toto/`. Export AnomalyXL into the layout the Toto eval
reads, then evaluate zero-shot or with a fine-tuned LoRA adapter:

```bash
uv run python anomalyXL/scripts/export_anomalyxl_for_toto.py \
    anomalyXL/datasets/anomalyxl-precise anomalyXL/datasets/toto-export/precise

cd sft/toto && uv sync
uv run python train.py --train-dir ... --val-dir ... --output-dir checkpoints/toto_ft_precise
uv run python eval.py --data ../../anomalyXL/datasets/toto-export/precise \
    --adapter checkpoints/toto_ft_precise/adapter --output-prefix results/toto_ft_precise
```

On the **precise** regime you must then run `score_precise.py`. The Toto eval's own
`is_correct` column is exact string equality against the full `category:{json}` gold —
the right metric for the coarse MCQ regime, but meaningless for precise, where the gold
is a JSON payload with exact integer endpoints. `score_precise.py` re-scores the model's
raw generation with the real AnomalyXL scorer (IoU / kind×IoU / composite):

```bash
uv run python score_precise.py results/toto_ft_precise_results.csv \
    --out results/toto_ft_precise_scored.csv
```

**Classical baseline (no LLM):** `anomalyXL/scripts/classical_baseline.py`. It
needs the `classical` extra (`cd anomalyXL && uv sync --extra classical`) — the
PELT change-point and matrix-profile features depend on `ruptures` and `stumpy`. Run
`features` once to build the feature cache, then `run` to fit one gradient-boosted head
per question category on the AnomalyXL train/val splits and score them with the same
scorers as every other row.

**Evaluating an RL checkpoint:** use `scripts/eval_v1_rlm.py`. It replays the exact
protocol the model was trained on (v1 prompt, 15-turn cap, the vendored `rlm` harness)
without the prime-rl orchestrator, and scores with the v0 scorer.

## 4. GRPO post-training

`configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml` is the config behind the paper's RL
row: full-parameter GRPO on the base `Qwen/Qwen3.5-4B` (no SFT warm start), trained
against the generator rather than a fixed corpus, so every batch is freshly synthesized
and no sample repeats within a run. The reward is the benchmark's own scorer plus a
group-aware token-expense penalty and a flat −0.25 for burning the 15-turn budget
without answering.

The run has **two phases**, and this matters: the generator's default 20% negatives are
a low-effort attractor from a cold start, and the policy collapses onto always-"no
anomaly". So negatives are withheld for the first 220 steps and re-introduced after, by
stopping and resuming from the step-220 checkpoint.

```bash
cd prime-rl

# Phase 1 — steps 0-220, negatives OFF.
export ANOMALYXL_P_PRESENT=1.0 ANOMALYXL_ALLCH_MIN_K=1
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync rl @ ../configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml

# Phase 2 — stop at step 220, unset the two vars (restoring the benchmark's 80/20
# split), and resume from the step-220 checkpoint.
unset ANOMALYXL_P_PRESENT ANOMALYXL_ALLCH_MIN_K
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync rl @ ../configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml
```

Both variables **must** be exported in the launch environment. Setting them in the
config's `harness.env` block does *not* work: that block reaches the rollout subprocess,
while `anomalyxl` is imported — and reads them, at module import — in the orchestrator's
env-server process, which only inherits the launch env. Setting them there silently
yields zero negatives.

### The inference servers

The config has no `[inference]` section: it expects **two standalone single-GPU vLLM
servers** at `localhost:8000` and `:8001`, started before the trainer. The tool-calling
flags are not optional — the RLM harness drives the model through tool calls, so a server
started without them never emits one and **every rollout scores zero**:

```bash
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$((i+2)) vllm serve Qwen/Qwen3.5-4B \
    --port $((8000+i)) \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 65536 --gpu-memory-utilization 0.85 &
done
```

Use `qwen3_coder`, **not** `qwen3_xml`: both parse simple calls, but `qwen3_xml`'s
streaming path mangles the JSON `arguments` of multi-line `code` payloads, which then
400s the following turn.

### Other launch notes

- **Always pass `uv run --no-sync`.** A plain `uv run` performs an implicit sync that
  prunes the trainer's optional `flash-attn` extra, and the next launch dies with
  `ModuleNotFoundError: No module named 'flash_attn'`.
- `scripts/setup.sh` clones **upstream** prime-rl at `main`. The paper's runs used a
  patched fork; pin your own with `PRIME_RL_REPO` / `PRIME_RL_REF` if `main` has drifted
  away from the `verifiers` v1 taskset/harness API this repo builds on.

You can smoke it for two steps before committing GPU-hours:

```bash
uv run --no-sync rl @ ../configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml --max-steps 2 \
  --trainer.wandb.offline true --orchestrator.wandb.offline true
```
