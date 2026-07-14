# AnomalyXL TSLM baselines — standalone SFT

Standalone, reproducible training + eval for the three TSLM baselines reported in
the TimeRLM paper — **ChatTS**, **OpenTSLM-Flamingo**, and **ITFormer** — all on a
frozen [Chronos-2](https://huggingface.co/amazon/chronos-2) time-series encoder
feeding a LoRA-tuned `Qwen/Qwen3.5-4B` backbone, trained/evaluated on **AnomalyXL**
in both its `precise` and `coarse` variants.

## Setup

```bash
cd sft/tslm
uv sync
```

This pulls a `transformers` build from git (rev `5c1c72b`, needed for Qwen3.5
support — no PyPI release supported it at the time these baselines were trained),
plus `chronos-forecasting`, `open-flamingo`, and `peft`.

## Layout

```
configs/                    6 YAML configs: {chatts,flamingo,itformer} x {precise,coarse}
src/models/
  chatts.py                 ChatTSModel (+ its MLPProjector)
  flamingo.py                OpenTSLMFlamingo (+ TimeSeriesFlamingoWithTrainableEncoder,
                              the Qwen3.5 attention-dispatcher monkeypatch)
  itformer.py                 ITFormerModel (+ ITFormerFusion, attention/positional-encoding
                              primitives, drop path) — everything ITFormer needs in one file
  chronos2_encoder.py        Chronos2TSEncoder — the one TS encoder all 3 architectures share
  qwen_backbone.py            QwenBackbone — the one LLM backbone all 3 architectures share
  base.py                     BaseModel — the abstract forward/generate/prepare_batch contract
                              that lets one train.py drive all 3 architectures without
                              per-architecture branching (kept; this one's load-bearing)
src/datasets/anomaly_xl/     AnomalyXLDataset + scoring.py (precise/coarse)
src/{prompt,training,utils}/ shared prompt dataclasses, W&B logger, config schema
train.py                     training entrypoint (raw PyTorch loop; config-driven)
aggregate_results.py         post-hoc TSV comparison table across all 6 runs
```

There's no registry/plugin-dispatch layer for datasets, encoders, projectors, or
backbones — `train.py` picks the model class from a 3-entry dict
(`MODEL_CLASSES = {"chatts": ..., "flamingo": ..., "itformer": ...}`), and each
architecture file imports `Chronos2TSEncoder`/`QwenBackbone` directly rather than
through a `get_encoder()`/`get_backbone()` lookup. That indirection made sense in
the original framework (which served many more datasets/architectures); here it
would just be dead generality.

## Data

Each config's `dataset.extra_kwargs.train_dir` points at
`anomalyXL/datasets/anomalyxl-train/anomalyxl-{precise,coarse}/{train,val}.parquet`
(relative to the repo root); the held-out gold test shard
(`anomalyXL/datasets/anomalyxl-{precise,coarse}/data.parquet`, 800/248 rows,
seed=42) is used automatically unless a config overrides `test_dir`.

Materialize all of it from the released Hub shards — `train` / `validation` / `test` map
onto `train.parquet` / `val.parquet` / `data.parquet`:

```bash
uv run python fetch_data.py          # both variants
```

Precise: 4488 train + 520 val = the 5008-row training shard the configs describe.

## Training

```bash
uv run python train.py --config configs/chatts_precise.yaml
```

Each run: loads Chronos-2 (frozen) + Qwen3.5-4B, injects LoRA (r=16, alpha=32) into
the Qwen decoder only, trains up to 5 epochs with early stopping (patience 3 on
val_loss), and automatically evaluates the best checkpoint against the gold test
shard at the end — results land under `results/<dataset>/<run_name>/`
(`summary.json`, `history.json`, `output_logs/test_epoch_*.json`,
`checkpoints/best_model.pt`). Add `--no-wandb` to disable W&B logging.

Architecture notes:
- **ChatTS** — LLaVA-style: Chronos-2 output → MLP projector → replaces `<ts_pad>`
  token positions in the LLM's input embeddings.
- **OpenTSLM-Flamingo** — Chronos-2 output → Perceiver Resampler (64 latents) →
  gated cross-attention layers interleaved into the Qwen decoder.
- **ITFormer** — multi-variable Chronos-2 (`GroupSelfAttention` across channels) →
  instruction-aware fusion (`ITFormerFusion`) → 25 learned prefix tokens merged
  into the LLM input at `<|image_pad|>` positions.

## Aggregating results

After running all 6 configs:

```bash
uv run python aggregate_results.py --results-dir results --out anomaly_xl_tslm_comparison.tsv
```

Produces one row per (architecture, variant) with per-category primary-metric
means, matching the paper's TSLM baseline table layout.
