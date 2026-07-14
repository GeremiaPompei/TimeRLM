#!/usr/bin/env bash
# Bootstrap a self-contained prime-rl training environment for TimeRLM.
#
# Installs prime-rl INTO the repo (./prime-rl, gitignored) and installs our three
# packages into prime-rl's venv, so that the ids used by the configs resolve:
#
#   environments/timeseries_qa     -> id "timeseries-qa"     (eval: text/image/rlm)
#   environments/timeseries_qa_v1  -> id "timeseries-qa-v1"  (RL taskset + harness)
#   anomalyXL               -> `anomalyxl` (the AnomalyXL generator)
#
# Assumes a Linux GPU machine with git, curl, and a CUDA toolchain available.
# For a fresh bare box you may instead run prime-rl's official installer first
# (it also handles apt deps + flash-attn builds):
#   curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
#
# Usage (from anywhere):
#   bash scripts/setup.sh
#
# Override the prime-rl source if you need a specific pin:
#   PRIME_RL_REPO=https://github.com/PrimeIntellect-ai/prime-rl.git \
#   PRIME_RL_REF=main bash scripts/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIME_RL_DIR="${REPO_ROOT}/prime-rl"
ENV_DIR="${REPO_ROOT}/environments/timeseries_qa"
ENV_V1_DIR="${REPO_ROOT}/environments/timeseries_qa_v1"
GEN_DIR="${REPO_ROOT}/anomalyXL"
PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
PRIME_RL_REF="${PRIME_RL_REF:-}"

echo "==> Repo root: ${REPO_ROOT}"

# 1. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# 2. Clone prime-rl into ./prime-rl (idempotent). Submodule URLs in .gitmodules
#    use SSH; rewrite to HTTPS on the fly so the public submodules clone without
#    SSH keys. The private submodules (research-environments, configs/private)
#    are optional for our use and skipped.
if [[ ! -d "${PRIME_RL_DIR}/.git" ]]; then
  echo "==> Cloning ${PRIME_RL_REPO} into ${PRIME_RL_DIR}"
  git clone "${PRIME_RL_REPO}" "${PRIME_RL_DIR}"
  if [[ -n "${PRIME_RL_REF}" ]]; then
    git -C "${PRIME_RL_DIR}" checkout "${PRIME_RL_REF}"
  fi
fi

cd "${PRIME_RL_DIR}"
# prime-rl's workspace declares editable deps on env packages under
# deps/research-environments (e.g. aime2024, math-env), so that submodule is
# required for `uv sync` even though we don't train on those envs. Only
# configs/private (research-configs) is genuinely private and skipped.
echo "==> Initializing required prime-rl submodules"
git -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init \
    deps/verifiers deps/renderers deps/pydantic-config deps/research-environments

# 3. Sync prime-rl's venv (trainer + inference + verifiers deps).
echo "==> uv sync --all-extras (this is the heavy step)"
uv sync --all-extras

# 4. Install our packages (editable) into prime-rl's venv. The generator and the
#    v1 taskset are required for RL; the v0 env is required for eval and for the
#    RL run's online eval.
echo "==> Installing anomalyxl + timeseries_qa + timeseries_qa_v1 into prime-rl venv"
# --no-deps on the env packages: `uv pip install -e` does NOT honor their
# [tool.uv.sources], so a plain install re-resolves `verifiers` from PyPI and
# clobbers the editable deps/verifiers submodule the RL path depends on.
uv pip install -e "${GEN_DIR}"
uv pip install --no-deps -e "${ENV_DIR}"
uv pip install --no-deps -e "${ENV_V1_DIR}"

# 5. Verify the ids resolve (id 'timeseries-qa' -> module 'timeseries_qa', etc).
echo "==> Verifying imports"
uv run python -c "import anomalyxl, timeseries_qa, timeseries_qa_v1; print('imports OK')"

# 6. A venv holding the vendored rlm harness, used by scripts/eval_v1_rlm.py to
#    replay the trained protocol without the prime-rl orchestrator.
echo "==> Creating rlm-harness venv"
uv venv "${REPO_ROOT}/rlm-harness/.venv"
uv pip install --python "${REPO_ROOT}/rlm-harness/.venv" -e "${REPO_ROOT}/rlm-harness"

echo
echo "==> Setup complete."
echo "    Eval and RL commands are in docs/REPRODUCE.md."
