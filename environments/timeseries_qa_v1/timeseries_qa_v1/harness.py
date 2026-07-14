"""RLM host-local harness (verifiers v1).

Runs the upstream `rlm` agent (persistent IPython kernel + auto-compaction + optional
sub-agents) host-local for the numeric AnomalyXL RLM path — via `rlm_program.py` as a uv
script in the subprocess runtime, not the upstream container harness. The taskset places
`context.json` and reads back `answer.json`; this harness just drives `rlm` against the
interception endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

# The vendored harness: <repo>/rlm-harness, resolved from this file's location so
# the checkout can live anywhere. `TIMERLM_RLM_HARNESS` overrides it (e.g. when the
# env is installed into prime-rl's venv from outside the repo tree).
_REPO_ROOT = Path(__file__).resolve().parents[3]
RLM_HARNESS = Path(os.environ.get("TIMERLM_RLM_HARNESS", _REPO_ROOT / "rlm-harness")).resolve()
if not (RLM_HARNESS / "pyproject.toml").exists():
    raise RuntimeError(
        f"rlm-harness not found at {RLM_HARNESS}. Point TIMERLM_RLM_HARNESS at the "
        "repo's rlm-harness/ directory."
    )

# `rlm_program.py` ships a placeholder in its PEP-723 dependency block; bind it to
# this checkout before the runtime resolves the script's deps.
PROGRAM_SOURCE = (Path(__file__).resolve().parent / "rlm_program.py").read_text().replace(
    "file://RLM_HARNESS_PATH", f"file://{RLM_HARNESS}"
)


class RLMLocalHarnessConfig(HarnessConfig):
    summarize_at_tokens: int = 0
    """Auto-compaction threshold (0 disables). Long multi-turn RLM trajectories benefit."""
    max_depth: int = 0
    """Recursive sub-agent depth (0 = no sub-agents)."""
    exec_timeout_s: int = 300


class RLMLocalHarness(Harness[RLMLocalHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # the task framing rides in the prompt; rlm owns its system prompt
    SUPPORTS_USER_SIM = False
    SUPPORTS_MESSAGE_PROMPT = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(trace.task)
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--prompt={prompt}",
            f"--summarize-at-tokens={self.config.summarize_at_tokens}",
            f"--max-depth={self.config.max_depth}",
            f"--exec-timeout={self.config.exec_timeout_s}",
        ]
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        return await runtime.run_program([*program, *args], {**self.config.env})
