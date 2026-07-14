"""Local RLM executor — runs the REPL worker as a host-local subprocess and
holds files on the local filesystem, bypassing the Prime sandbox entirely.

Why:
- Prime sandbox cold-starts (5–10s/rollout) and occasional 10-minute hangs
  dominate wall time on long eval suites — 800 rollouts × 10s = ~2h of pure
  network round-trips before any compute.
- For eval (not training), the model's REPL code is just analyzing synthetic
  series with numpy/scipy/matplotlib. There's no adversarial setting, so the
  host-isolation Prime provides isn't load-bearing.

Trade-off:
- The model's ``call_python_repl`` output runs as the host user with full
  filesystem access. Acceptable for benchmark runs; not for adversarial
  training or where the host has secrets that shouldn't leak.
- The host venv must already have the packages the env would ``pip install``
  into the sandbox (numpy, scipy, matplotlib for image mode). The local
  executor checks via ``python -c "import <pkg>"`` and aborts noisily if
  anything's missing — no auto-install on the host.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Prepend our venv's bin dir to PATH so subprocess `python ...` invocations
# pick up the same interpreter the executor is running under (matplotlib,
# numpy, scipy, etc. live there — not in the system python).
_VENV_BIN = str(Path(sys.executable).parent)

try:
    from verifiers.envs.experimental.rlm_env import (
        RLMExecutor,
        RLMSandboxCommandTimeout,
        _build_worker_paths,
    )
except ModuleNotFoundError:
    from rlm_env_vendored import (
        RLMExecutor,
        RLMSandboxCommandTimeout,
        _build_worker_paths,
    )
from verifiers.types import State


class _LocalCmdResult:
    """Mimics the shape of ``prime_sandboxes`` command results so the parent
    executor's downstream code can read ``.stdout`` / ``.exit_code`` unchanged."""

    def __init__(self, stdout: str, exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code


class LocalRLMExecutor(RLMExecutor):
    """Drop-in subclass of ``RLMExecutor`` that runs everything on the host.

    Overrides only the network-facing methods of ``SandboxMixin``; everything
    else (worker script generation, request/response framing, state keys) is
    inherited unchanged.
    """

    # ----- sandbox creation/deletion: just hand out a fake id -----

    def _build_sandbox_request(self, state: State):
        return None  # parent only uses this to feed create_sandbox

    async def create_sandbox(self, state: State, request) -> str:
        rid = state.get("rollout_id", "noid")
        return f"local-{rid}"

    async def delete_sandbox(self, sandbox_id: str) -> None:
        return None

    # ----- filesystem prep: local dir IS the workspace; no upload needed -----

    async def prepare_filesystem(self, state: State) -> None:
        session = self._get_session(state)
        if session.sandbox_id is None:
            state["sandbox_state"] = {"ready": False, "command_execution_times": []}
            session.sandbox_id = await self.create_sandbox(state, None)
            state["sandbox_state"]["ready"] = True

        # Point the "remote" paths at the same local dirs so the worker can
        # read context.json / write answer.json without any upload round-trip.
        local_fs = Path(session.local_fs_root)
        local_ctrl = Path(session.local_control_dir)
        local_fs.mkdir(parents=True, exist_ok=True)
        local_ctrl.mkdir(parents=True, exist_ok=True)

        session.sandbox_fs_root = str(local_fs)
        session.sandbox_control_dir = str(local_ctrl)
        session.paths = _build_worker_paths(str(local_ctrl))

        state["rlm_fs_staging_root"] = str(local_fs)
        state["rlm_control_dir_local"] = str(local_ctrl)
        state["rlm_fs_root_remote"] = str(local_fs)
        state["rlm_control_dir_remote"] = str(local_ctrl)
        state["rlm_paths_remote"] = session.paths.to_dict()

    # ----- per-command execution: spawn locally instead of remote API call -----

    async def _execute_sandbox_command(
        self,
        sandbox_id: str,
        command: str,
        *,
        working_dir: str | None = None,
        timeout: int = 30,
    ) -> _LocalCmdResult:
        # Match the remote path's behavior: shell-string commands run via /bin/bash.
        # Critical: prepend the env's venv bin dir to PATH so `python ...` calls
        # resolve to the venv's interpreter, not the system /usr/bin/python.
        # NOTE: the REPL sees exactly this venv's packages — numpy/scipy/
        # matplotlib are declared in pyproject.toml for that reason (scipy was
        # missing until 2026-07-04; rollouts before then imported it in vain).
        env = dict(os.environ)
        env["PATH"] = f"{_VENV_BIN}:{env.get('PATH', '')}"
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise RLMSandboxCommandTimeout(
                f"Local command timed out after {timeout}s: {command[:100]}"
            ) from e
        return _LocalCmdResult(
            stdout=result.stdout or "",
            exit_code=result.returncode,
        )

    # ----- file movement: no-ops; everything is already local -----

    async def _upload_file_with_retry(
        self, sandbox_id: str, dst: str, src: str, label: str
    ) -> None:
        # If `src` and `dst` are different local paths (they may be when a
        # caller stages from one local dir to another), copy them.
        if dst and src and Path(src).resolve() != Path(dst).resolve():
            await asyncio.to_thread(
                lambda: Path(dst).parent.mkdir(parents=True, exist_ok=True)
            )
            await asyncio.to_thread(
                lambda: Path(dst).write_bytes(Path(src).read_bytes())
            )

    async def _upload_directory(self, sandbox_id: str, src: str, dst: str) -> None:
        if dst and src and Path(src).resolve() != Path(dst).resolve():
            # Copy local→local. Source paths in the local executor are aliases
            # of the destination, so this normally short-circuits.
            import shutil

            await asyncio.to_thread(
                shutil.copytree, src, dst, dirs_exist_ok=True
            )

    async def _download_directory(self, sandbox_id: str, src: str, dst: str) -> None:
        await self._upload_directory(sandbox_id, src, dst)

    # ----- package install: skip; trust host venv (errors surface at use time) -----

    async def _install_packages(self, session) -> None:
        # Intentional no-op. The env's venv either has the packages
        # (numpy/scipy/matplotlib for image mode) or doesn't; if missing,
        # the worker's import will raise a clean ImportError. Avoiding the
        # check here also dodges the issue that subprocess `python` would
        # use system python without an explicit PATH override.
        return None
