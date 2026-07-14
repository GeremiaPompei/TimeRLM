# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
#   "rlm @ file://RLM_HARNESS_PATH",  # vendored rlm-harness (see rlm-harness/NOTICE)
# ]
# ///
"""Runs the vendored `rlm` agent host-local for one rollout.

`RLM_HARNESS_PATH` in the dependency block above is a placeholder: `harness.py`
substitutes the absolute path of the repo's `rlm-harness/` (override with the
`TIMERLM_RLM_HARNESS` env var) before handing this source to the runtime, so the
script stays portable across checkouts.

The harness launches this as a uv script in the (subprocess) runtime, so the IPython
kernel `rlm` drives runs on the host in the runtime workspace — where the taskset has
placed `context.json`. The model's calls go to the interception endpoint (the v1
framework records them into the trace for training). The secret arrives as argv and is
used to build the OpenAI client *in-process*; it is never put in the environment, so
the IPython kernel `rlm` spawns cannot read it. The model writes `answer.json`; the
taskset scores it.
"""

import argparse
import asyncio
import os

import rlm
from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--team-id", default=None,
                    help="sent as X-Prime-Team-ID; Prime Inference rejects team-scoped "
                         "accounts' keys without it")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--summarize-at-tokens", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=0)
    p.add_argument("--exec-timeout", type=int, default=300)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    # Non-secret knobs the engine reads from env (the kernel may inherit these; fine).
    os.environ["RLM_MODEL"] = args.model
    max_turns = int(os.environ.get("RLM_MAX_TURNS", "0"))
    if max_turns > 0:
        # Surface the turn budget in the task prompt so the policy can plan
        # ("try hard, fail fast") instead of discovering the cap at the end.
        args.prompt += (
            f"\n\nYou have a hard budget of {max_turns} turns (each tool call"
            " counts as one). On the final turn tools are disabled and you must"
            " state your final answer; unanswered work scores zero."
        )
    os.environ["RLM_MAX_DEPTH"] = str(args.max_depth)
    os.environ["RLM_EXEC_TIMEOUT"] = str(args.exec_timeout)
    if args.summarize_at_tokens > 0:
        os.environ["RLM_SUMMARIZE_AT_TOKENS"] = str(args.summarize_at_tokens)

    # The secret stays out of the environment: the client is built here and passed in.
    headers = {"X-Prime-Team-ID": args.team_id} if args.team_id else {}
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, default_headers=headers)
    result = await rlm.run(args.prompt, client=client, cwd=os.getcwd())
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
