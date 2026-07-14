#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash autodl/setup.sh
# Optional overrides:
#   ENV_DIR=/root/autodl-tmp/envs/medvlm-grpo \
#   CACHE_DIR=/root/autodl-tmp/cache \
#   PYTHON_BIN=python3.11 bash autodl/setup.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DISK="${AUTODL_DATA_DISK:-/root/autodl-tmp}"
if [[ -d "${DATA_DISK}" ]]; then
  DEFAULT_ENV_DIR="${DATA_DISK}/envs/medvlm-grpo"
  DEFAULT_CACHE_DIR="${DATA_DISK}/cache"
else
  DEFAULT_ENV_DIR="${PROJECT_ROOT}/.venv-autodl"
  DEFAULT_CACHE_DIR="${HOME}/.cache"
fi

ENV_DIR="${ENV_DIR:-${DEFAULT_ENV_DIR}}"
CACHE_DIR="${CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; select an AutoDL NVIDIA GPU image."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} not found. Set PYTHON_BIN to Python 3.10-3.12."

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Python 3.10-3.12 is required, found {sys.version.split()[0]}")
print(f"Python: {sys.version.split()[0]}")
PY

echo "NVIDIA GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

mkdir -p "${ENV_DIR}" "${CACHE_DIR}/huggingface" "${CACHE_DIR}/uv" "${CACHE_DIR}/torch_extensions"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip uv

export HF_HOME="${CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export UV_LINK_MODE=copy
export TORCH_EXTENSIONS_DIR="${CACHE_DIR}/torch_extensions"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Installing the resolved CUDA training stack. This can take several minutes..."
"${ENV_DIR}/bin/uv" pip install \
  --python "${PYTHON}" \
  --torch-backend=auto \
  -r "${PROJECT_ROOT}/autodl/requirements.txt" \
  -e "${PROJECT_ROOT}"

echo "Checking dependency consistency..."
"${PYTHON}" -m pip check
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_environment.py" --require-unsloth

cat <<EOF

Environment installation completed.

Activate it in each new terminal with:
  source "${PROJECT_ROOT}/autodl/activate.sh"

Then run a short GRPO smoke job with:
  python scripts/train_unsloth_grpo.py --max_steps 2 --output_dir output/autodl-smoke
EOF
