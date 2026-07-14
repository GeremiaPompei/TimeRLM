#!/usr/bin/env python
"""LoRA SFT of Toto-1.0-QA-Experimental on AnomalyXL, protocol-matched to the
TSLM baselines (chatts/flamingo/itformer, see ../tslm/):

  epochs=5, lr=1e-4, batch_size=1, AdamW(0.9,0.999,eps=1e-8), weight_decay=0.01,
  max_grad_norm=1.0, linear schedule with 3% warmup, early stopping patience=3
  on val loss, LoRA r=16 alpha=32 dropout=0.05 on the text decoder.

Architecture-specific glue (allowed to differ per baseline): Toto keeps its own
TS encoder + projector (frozen, like the baselines' frozen Chronos-2) and its
native image+TS input pipeline; supervision target is the raw gold answer
string, as in the baseline QADatasets.

Data: exports already sitting at ../../anomalyXL/datasets/toto-export/
(benchmark.csv + arfbench-ts-data/ + arfbench-images/ per split).

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python train.py \\
      --train-dir ../../anomalyXL/datasets/toto-export/ft-coarse-train \\
      --val-dir   ../../anomalyXL/datasets/toto-export/ft-coarse-val \\
      --output-dir checkpoints/toto_ft_coarse
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation.eval_toto_vlm import (  # noqa: E402
    get_model_prompt,
    load_model_and_processor,
    load_timeseries_adaptive,
    move_ts_data_to_device,
    select_max_ts_groups_for_task,
)

# Baseline protocol constants — keep in sync with the TSLM configs (../tslm/configs/).
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.03
PATIENCE = 3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05


def build_items(data_dir: str, max_ts_length: int, ts_interval: int):
    """Materialize (question, image_paths, ts_data, target) items for a split."""
    root = Path(data_dir)
    df = pd.read_csv(root / "benchmark.csv")
    items = []
    for _, row in df.iterrows():
        qg = str(row["query_group"])
        image = root / "arfbench-images" / f"{qg}.png"
        if not image.exists():
            continue
        max_groups = select_max_ts_groups_for_task(row.get("task_category"), 100, 1000)
        ts_data = load_timeseries_adaptive(
            qg, str(root / "arfbench-ts-data"),
            max_length=max_ts_length, max_groups=max_groups,
            auto_interval=(ts_interval == 0),
            fixed_interval=ts_interval or None,
        )
        options = str(row["options_str"]).replace('"', "")
        question = f"{row['question']}\nAnswer Choices: {options}"
        items.append({
            "question": question,
            "image_paths": [str(image)],
            "ts_data": ts_data,
            "target": str(row["correct_answer"]),
        })
    return items


def encode_example(item, processor, device):
    """Assemble inputs exactly like eval, then append the supervised target."""
    text_prompt, images, _ = get_model_prompt(
        item["question"], item["image_paths"], processor, ts_data=item["ts_data"],
    )
    target = item["target"] + processor.tokenizer.eos_token
    prompt_ids = processor(
        text=[text_prompt], images=[images], return_tensors="pt",
        padding=False, truncation=False,
    )
    target_ids = processor.tokenizer(
        target, return_tensors="pt", add_special_tokens=False
    ).input_ids

    input_ids = torch.cat([prompt_ids["input_ids"], target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : prompt_ids["input_ids"].size(1)] = -100  # supervise answer only

    batch = {k: v for k, v in prompt_ids.items() if k not in ("input_ids", "attention_mask")}
    batch.update(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    move_ts_data_to_device(item["ts_data"], device)
    return batch


def epoch_pass(model, items, processor, device, optimizer=None, scheduler=None,
               start_at=0, on_ckpt=None):
    total, n = 0.0, 0
    for i, item in enumerate(items):
        if i < start_at:
            continue
        batch = encode_example(item, processor, device)
        out = model(ts_data=[item["ts_data"]] if item["ts_data"] is not None else None, **batch)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total += float(loss.detach()); n += 1
        # move_ts_data_to_device mutates the cached item IN PLACE onto CUDA;
        # move it back or all N samples' tensors accumulate on the GPU across
        # the epoch (observed: 80GB -> 94GB creep -> OOM at step ~1600).
        move_ts_data_to_device(item["ts_data"], "cpu")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)} loss={total/n:.4f}", flush=True)
        if on_ckpt is not None and (i + 1) % 500 == 0:
            on_ckpt(i + 1)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="Datadog/Toto-1.0-QA-Experimental")
    ap.add_argument("--max-ts-length", type=int, default=12800)
    ap.add_argument("--ts-interval", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    model, processor = load_model_and_processor(args.model_path)
    device = next(model.parameters()).device

    # Freeze everything, then LoRA the text decoder (protocol: encoder frozen,
    # LoRA-LLM only). Their VLM lives at model.vlm; language part per Qwen3-VL.
    for p in model.parameters():
        p.requires_grad = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model.vlm = get_peft_model(model.vlm, lora)
    model.vlm.print_trainable_parameters()
    if hasattr(model.vlm, "enable_input_require_grads"):
        model.vlm.enable_input_require_grads()
    if hasattr(model.vlm, "gradient_checkpointing_enable"):
        model.vlm.gradient_checkpointing_enable()

    print("loading data...", flush=True)
    train_items = build_items(args.train_dir, args.max_ts_length, args.ts_interval)
    val_items = build_items(args.val_dir, args.max_ts_length, args.ts_interval)
    print(f"train={len(train_items)} val={len(val_items)}", flush=True)

    total_steps = EPOCHS * len(train_items)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    history, best_val, bad_epochs = [], math.inf, 0

    # Crash-resume: long runs can die opaquely — persist adapter+optimizer+scheduler
    # every 500 steps.
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    state_p = out / "train_state.pt"
    start_epoch, start_step = 1, 0
    if state_p.exists():
        st = torch.load(state_p, map_location="cpu")
        set_peft_model_state_dict(model.vlm, st["adapter"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        history, best_val, bad_epochs = st["history"], st["best_val"], st["bad_epochs"]
        start_epoch, start_step = st["epoch"], st["step"]
        print(f"resumed from epoch {start_epoch} step {start_step}", flush=True)
    elif (out / "adapter").exists() and (out / "history.json").exists():
        # Legacy fallback: a pre-checkpointing run died after completing >=1
        # epoch. Recover the best adapter + history; the linear schedule is a
        # pure function of step count, so fast-forward it. Adam moments are
        # lost (re-warm within tens of steps — negligible mid-training).
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.vlm, load_file(out / "adapter" / "adapter_model.safetensors"))
        history = json.loads((out / "history.json").read_text())
        best_val = min(h["val_loss"] for h in history)
        start_epoch = len(history) + 1
        for _ in range(len(train_items) * len(history)):
            scheduler.step()
        print(f"legacy-resumed at epoch {start_epoch} (adapter + history, fresh optimizer)", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        def on_ckpt(step, _epoch=epoch):
            torch.save({
                "adapter": get_peft_model_state_dict(model.vlm),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history, "best_val": best_val, "bad_epochs": bad_epochs,
                "epoch": _epoch, "step": step,
            }, state_p)
        model.train()
        tr = epoch_pass(model, train_items, processor, device, optimizer, scheduler,
                        start_at=start_step if epoch == start_epoch else 0, on_ckpt=on_ckpt)
        start_step = 0
        model.eval()
        with torch.no_grad():
            vl = epoch_pass(model, val_items, processor, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl})
        print(f"epoch {epoch}: train={tr:.4f} val={vl:.4f}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        # end-of-epoch state: resume lands at the NEXT epoch, step 0
        torch.save({
            "adapter": get_peft_model_state_dict(model.vlm),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history, "best_val": best_val, "bad_epochs": bad_epochs,
            "epoch": epoch + 1, "step": 0,
        }, state_p)
        if vl < best_val:
            best_val, bad_epochs = vl, 0
            model.vlm.save_pretrained(out / "adapter")
            print("  saved best adapter", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stop", flush=True)
                break
    print("done. best val:", best_val, flush=True)


if __name__ == "__main__":
    main()
