# Toto baseline — standalone SFT + eval

Standalone, reproducible fine-tuning + evaluation for the paper's Toto baseline —
`Datadog/Toto-1.0-QA-Experimental` (a frozen `Datadog/Toto-Open-Base-1.0` time-series
encoder + LoRA-tuned `Qwen/Qwen3-VL-32B-Instruct` vision-language model) — on
**AnomalyXL** in both its `precise` and `coarse` variants.

## Provenance

This pipeline was originally built and run in a separate codebase, whose two
fine-tuning runs produced the Toto (SFT) rows in the paper's main results table.
The recipe is protocol-matched to the paper's TSLM baselines (see `../tslm/`):
same epochs/LR/weight-decay/warmup/patience/batch-size, LoRA on the text decoder
only (r=16, alpha=32, dropout=0.05) with the time-series encoder frozen — applied
here to Toto+Qwen3-VL-32B instead of Chronos-2+Qwen3.5-4B. This directory is a
trimmed, standalone port of that code. No external checkout or dependency
remains — everything here resolves relative to this repo.

Training already completed once (see the paper's tables for the resulting
numbers); this port exists to make that recipe reproducible from this repo
alone. Only the code is shipped here — no trained weights. Run `train.py` to
reproduce the two adapters (`checkpoints/toto_ft_{coarse,precise}/adapter/`,
gitignored) before running fine-tuned eval.

## Setup

```bash
cd sft/toto
uv sync
```

Pulls exact-pinned deps: `torch==2.8.0`, `transformers==4.57.1` (stock PyPI —
Qwen3-VL support has been upstream since 4.57, no git build needed, unlike
`../tslm/`'s Qwen3.5 pin), `peft==0.18.1`, `toto-ts==0.2.0` (the Toto encoder
package), `qwen-vl-utils==0.0.14`, `torchvision==0.23.0`. These are pinned
exactly rather than loose-bounded: the model code monkeypatches
`transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_placeholder_mask`
at import time (so Qwen3-VL's image-token detection skips the time-series
embedding positions prepended to `inputs_embeds`) — a newer transformers release
could silently rename or change that method's signature and break the patch.

## Layout

```
train.py                     training entrypoint (raw PyTorch loop; same CLI as the
                              original recipe)
eval.py                      eval entrypoint — omit --adapter for zero-shot (base
                              model), pass it to evaluate a fine-tuned adapter
checkpoints/                  gitignored — produced locally by train.py, not shipped
src/
  model/
    toto_vlm_components.py    TimeSeriesData, TotoEmbeddingExtractor, TotoVariateEmbedding,
                              TimeSeriesProjector, TotoAnomalyQAModel, and the
                              Qwen3-VL get_placeholder_mask monkeypatch
    load_checkpoint.py         Hub/mixin-format + legacy-adapter checkpoint loading,
                              single-GPU and tensor-parallel variants
  evaluation/
    eval_toto_vlm.py           reusable eval library: interval selection, prompt
                              assembly, generation, scoring loop
  utils/
    inference_utils.py         parse_response (answer/reasoning extraction),
                              get_image_paths
    compute_statistics.py      accuracy/F1/tier-weighted-F1 scoring (kept as-is —
                              self-contained binning logic; correct for `coarse`)
    log_utils.py                incremental-result saving + progress logging
    prompt_test_benchmark.py    the exact prompt template (must not drift)
    anomalyxl_scoring.py        the real AnomalyXL precise scorer (IoU / kind-match /
                              composite), vendored from ../tslm/ — see score_precise.py
score_precise.py              re-scores a precise results CSV with anomalyxl_scoring.py
```

Each export dir (`benchmark.csv` + `arfbench-ts-data/` + `arfbench-images/`) is
already sitting at `../../anomalyXL/datasets/toto-export/{coarse,precise,
ft-coarse-train,ft-coarse-val,ft-precise-train,ft-precise-val}/` — nothing needs
to be regenerated. `train.py`/`eval.py` take these as plain `--train-dir`/
`--val-dir`/`--data` CLI args, same as the original recipe (no YAML config layer
needed here, unlike `../tslm/` — these scripts were already simple, path-agnostic
CLIs upstream).

## Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --train-dir ../../anomalyXL/datasets/toto-export/ft-coarse-train \
    --val-dir   ../../anomalyXL/datasets/toto-export/ft-coarse-val \
    --output-dir checkpoints/toto_ft_coarse_rerun
```

5 epochs, LoRA (r=16, alpha=32) on `model.vlm` only, batch size 1, crash-resume
every 500 steps to `train_state.pt`. Best-val adapter is saved to
`<output-dir>/adapter/`. Single GPU (~96GB-class) — matches how this was
originally trained despite the model card's "4x A100" minimum; no quantization.

## Evaluation

```bash
# Zero-shot (base model):
CUDA_VISIBLE_DEVICES=0 uv run python eval.py \
    --data ../../anomalyXL/datasets/toto-export/coarse \
    --output-prefix results/toto_zeroshot_coarse

# Fine-tuned (merges a LoRA adapter produced by train.py before inference):
CUDA_VISIBLE_DEVICES=0 uv run python eval.py \
    --data ../../anomalyXL/datasets/toto-export/precise \
    --adapter checkpoints/toto_ft_precise/adapter \
    --output-prefix results/toto_ft_precise_eval
```

Greedy decoding (temperature 0), incremental CSV checkpointing (resumable),
exact-match scoring against `correct_answer`. Writes four files:
`{prefix}_results.csv` (per-item, includes `prediction_raw` — the model's
undecoded generation), `{prefix}_statistics.csv` (accuracy by category/tier),
`{prefix}_f1.csv` (multiclass F1 per category), `{prefix}_tier_f1.csv`
(tier-weighted F1). Pass `--limit N` to smoke-test on the first N rows.

**`coarse` is scored correctly by the above** — it's MCQ, exact-match against
`correct_answer` is the right metric. **`precise` is not**: its gold is a JSON
payload with exact integer endpoints (`localize:{"_L": 16384, "end": 9751,
"present": true, "start": 9556}`), so exact-match on the full string is ~always
false regardless of how close the model's answer is — `is_correct`/`{prefix}
_statistics.csv` are not meaningful for `precise`. Re-score those with:

```bash
uv run python score_precise.py results/toto_ft_precise_eval_results.csv \
    --out results/toto_ft_precise_scored.csv
```

which runs each row's `prediction_raw` through the real AnomalyXL precise
scorer (IoU / kind-match / composite per category — the same scorer TimeRLM,
the TSLM baselines, and the classical baseline all use), matching how the
paper's Toto precise numbers were actually produced.

## GPU / compute

Single GPU, no quantization, gradient checkpointing on the VLM, batch size 1 —
matches how this was originally trained and evaluated. `Qwen/Qwen3-VL-32B-Instruct`
is ~65GB in bf16 alone; point `CUDA_VISIBLE_DEVICES` at a free ~96GB-class GPU
before launching.
