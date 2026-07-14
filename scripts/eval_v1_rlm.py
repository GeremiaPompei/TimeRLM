#!/usr/bin/env python3
"""Standalone v1-protocol eval: run a checkpoint (or any OpenAI-compatible
remote model) through the SAME interface RL checkpoints were trained on
(rlm-harness IPython agent, v1 prompt) without the prime-rl orchestrator, and
score with the v0 AnomalyXL scorer.

Protocol pieces mirrored 1:1 from the v4 training setup:
- prompt: timeseries_qa_v1.taskset._build_note + question  (copied verbatim)
- rollout: environments/timeseries_qa_v1/timeseries_qa_v1/rlm_program.py run as
  a subprocess in a per-task workspace holding context.json (subprocess runtime)
- harness env (configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml): RLM_MAX_TOKENS=24576,
  RLM_MAX_OUTPUT=8192, RLM_MAX_TURNS=15 (--max-turns, 0 = uncapped),
  summarize_at_tokens=45000, exec_timeout=300, max_depth=0
- reward: ds.score_answer(final answer, gold)  (== v0 `correct_answer`)

Run with the v0 env venv (needs ts_datasets):
  environments/timeseries_qa/.venv/bin/python scripts/eval_v1_rlm.py \
      --dataset anomalyxl --repo-or-path .../anomalyxl-precise --eval-size 800 \
      --base-url http://localhost:8012/v1 --model qwen4b-rl-v4-step145 \
      --out outputs/evals/v1-protocol/v4-step145-precise.jsonl
Resumable: rows already in --out are skipped.

Against a remote provider (e.g. Prime Inference), omit --api-key to pick it up
from RLM_API_KEY / PRIME_API_KEY / OPENAI_API_KEY (in that order), and pass
--max-turns 0 for an uncapped run:
  environments/timeseries_qa/.venv/bin/python scripts/eval_v1_rlm.py \
      --dataset anomalyxl --repo-or-path .../anomalyxl-precise --eval-size 800 \
      --base-url https://api.pinference.ai/api/v1 --model z-ai/glm-5.2 \
      --max-turns 0 --out outputs/evals/v1-protocol/glm-5.2-precise.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/timeseries_qa"))
from ts_datasets import DATASETS  # noqa: E402

# Interpreter of a venv with the vendored rlm-harness installed. scripts/setup.sh
# creates it; override with RLM_PYTHON if it lives elsewhere.
RLM_PYTHON = Path(os.environ.get("RLM_PYTHON", ROOT / "rlm-harness/.venv/bin/python"))
RLM_PROGRAM = ROOT / "environments/timeseries_qa_v1/timeseries_qa_v1/rlm_program.py"

# configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml [orchestrator.train.env] harness block,
# minus RLM_MAX_TOKENS: the training-time completion budget is lifted for eval
# (user call — no caps inside the reasoning band). RLM_MAX_TURNS is a CLI flag
# (--max-turns), not hardcoded, since it's only meaningful as a training-fidelity
# knob for RL-checkpoint replay and shouldn't silently apply to other models.
HARNESS_ENV = {"RLM_MAX_OUTPUT": "8192"}
SUMMARIZE_AT_TOKENS = 45000
EXEC_TIMEOUT_S = 300
ROLLOUT_WALL_TIMEOUT_S = 2400  # backstop; RLM_MAX_TOKENS bounds normal rollouts

# Never let these reach the subprocess env: rlm_program.py takes the API key
# only via argv and builds the OpenAI client in-process specifically so the
# IPython kernel it spawns (which inherits this env) cannot read the secret.
_SECRET_ENV_KEYS = ("RLM_API_KEY", "PRIME_API_KEY", "OPENAI_API_KEY")


def _build_note(values_note: str) -> str:
    # verbatim from environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py
    return (
        "This is a critical application: when relevant, precisely locate the anomaly's "
        "start and end using statistical thresholds — be precise. The time series is in "
        'the file `context.json` in your working directory, shaped as '
        '{"series": {channel: [values], ...}}; load it with json. '
        + values_note
        + "Always read the actual channel keys from the data (do not guess names). "
        "Analyze it with numpy and scipy in the IPython tool. When you are done, stop "
        "calling the tool and state your final answer as a single JSON object matching "
        'the question\'s schema exactly (e.g. {"start": 12, "end": 87}), with no prose '
        "and no markdown fences. Format failures score zero."
    )


async def run_rollout(idx, prompt, series, gold, ds, args, ws_root, sem, out_f, lock, stats):
    async with sem:
        ws = ws_root / f"{idx:04d}"
        ws.mkdir(parents=True, exist_ok=True)
        # to_thread: a 131k x 16ch series dumps to ~35MB JSON — keep it off the loop
        await asyncio.to_thread(
            (ws / "context.json").write_text, json.dumps({"series": series}))
        env = {
            **{k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS},
            **HARNESS_ENV,
            "RLM_MAX_TURNS": str(args.max_turns),
            "RLM_SESSION_DIR": str(ws / "session"),  # keeps messages.jsonl with the row
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
        }
        t0 = time.time()
        # File-backed stdio, not PIPE: the rlm program's ipykernel grandchildren
        # inherit stdio FDs, and communicate() blocks on pipe EOF until every
        # inheritor exits — lingering kernels wedged whole runs. wait() only
        # needs the direct child to exit.
        out_log, err_log = ws / "stdout.txt", ws / "stderr.txt"
        with open(out_log, "wb") as so, open(err_log, "wb") as se:
            team_arg = [f"--team-id={args.team_id}"] if args.team_id else []
            proc = await asyncio.create_subprocess_exec(
                str(RLM_PYTHON), str(RLM_PROGRAM),
                f"--base-url={args.base_url}", f"--api-key={args.api_key}", *team_arg,
                f"--model={args.model}", f"--prompt={prompt}",
                f"--summarize-at-tokens={SUMMARIZE_AT_TOKENS}",
                "--max-depth=0", f"--exec-timeout={EXEC_TIMEOUT_S}",
                cwd=str(ws), env=env, stdout=so, stderr=se,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=ROLLOUT_WALL_TIMEOUT_S)
                answer = out_log.read_text(errors="replace").strip()
                error = None if proc.returncode == 0 else (
                    f"exit={proc.returncode}: {err_log.read_text(errors='replace')[-800:]}")
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                answer, error = "", f"wall timeout {ROLLOUT_WALL_TIMEOUT_S}s"
        # Malformed answers (e.g. {"start": null}) crash the scorer with
        # TypeError; the protocol says format failures score zero.
        try:
            reward = ds.score_answer(answer, gold) if answer else 0.0
        except Exception as e:
            reward = 0.0
            error = (error or "") + f" [scorer: {type(e).__name__}: {e}]"
        try:
            metrics = ds.score_metrics(answer, gold) if answer else {}
        except Exception:
            metrics = {}
        num_turns = None
        mpath = ws / "session" / "messages.jsonl"
        if mpath.exists():
            turns = [json.loads(l).get("turn", -1) for l in mpath.read_text().splitlines()
                     if l.strip()]
            num_turns = (max(turns) + 1) if turns else 0
        row = {
            "example_id": idx, "answer": gold, "model_answer": answer[-2000:],
            "reward": reward, "metrics": metrics, "error": error,
            "num_turns": num_turns,
            "seconds": round(time.time() - t0, 1),
        }
        async with lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            stats["n"] += 1
            stats["sum"] += reward
            if num_turns is not None:
                stats["turns_sum"] += num_turns
                stats["turns_n"] += 1
            mean_turns = stats["turns_sum"] / max(stats["turns_n"], 1)
            print(f"[{stats['n']}/{stats['total']}] id={idx} reward={reward:.3f} "
                  f"avg={stats['sum']/stats['n']:.4f} turns={num_turns} "
                  f"mean_turns={mean_turns:.1f} ({row['seconds']}s)"
                  + (f" ERR: {error[:120]}" if error else ""), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo-or-path", default=None)
    ap.add_argument("--eval-size", type=int, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=None,
                     help="defaults to RLM_API_KEY / PRIME_API_KEY / OPENAI_API_KEY env, in that order")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-turns", type=int, default=15,
                     help="0 = uncapped (matches engine.py: RLM_MAX_TURNS<=0 means no limit). "
                          "Default 15 preserves the v4 training-time turn budget for RL-checkpoint replay.")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (smoke)")
    args = ap.parse_args()

    # Mirrors verifiers' client_utils._build_headers_and_api_key: fall back to
    # the Prime CLI's cached login (`prime login`) when no key/team env var is set.
    prime_config = {}
    prime_config_path = Path.home() / ".prime" / "config.json"
    if prime_config_path.exists():
        try:
            prime_config = json.loads(prime_config_path.read_text())
        except json.JSONDecodeError:
            prime_config = {}
    if args.api_key is None:
        args.api_key = (os.environ.get("RLM_API_KEY") or os.environ.get("PRIME_API_KEY")
                         or os.environ.get("OPENAI_API_KEY") or prime_config.get("api_key"))
    if not args.api_key:
        raise SystemExit(
            "no API key: pass --api-key or set RLM_API_KEY / PRIME_API_KEY / OPENAI_API_KEY, "
            "or `prime login`")
    args.team_id = os.environ.get("PRIME_TEAM_ID") or prime_config.get("team_id")

    cls = DATASETS[args.dataset]
    kwargs = {"eval_size": args.eval_size, "mode": "rlm"}
    if args.repo_or_path:
        kwargs["repo_or_path"] = args.repo_or_path
    ds = cls(**kwargs)
    frame = ds.load_split("eval")  # df.iloc[:eval_size] — same rows/order as vf-eval
    note = _build_note(ds.rlm_values_note)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    ws_root = out_path.with_suffix("") / "workspaces"

    items = []
    for i, row in enumerate(frame):
        if args.limit and i >= args.limit:
            break
        if i in done:
            continue
        sample = ds.to_sample(row)
        items.append((i, f"{note}\n\n{sample['question']}", sample["series"], sample["answer"]))
    print(f"{len(items)} rollouts to run ({len(done)} already done) -> {out_path}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"n": 0, "sum": 0.0, "turns_sum": 0, "turns_n": 0, "total": len(items)}
    with open(out_path, "a") as out_f:
        results = await asyncio.gather(*(
            run_rollout(i, p, s, g, ds, args, ws_root, sem, out_f, lock, stats)
            for i, p, s, g in items), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            print(f"ROLLOUT EXCEPTION: {type(r).__name__}: {r}", flush=True)
    if stats["n"]:
        print(f"DONE {out_path}: n={stats['n']} avg_reward={stats['sum']/stats['n']:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
