#!/usr/bin/env bash

# This file must be sourced, not executed:
#   source autodl/activate.sh

_MEDVLM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_MEDVLM_DATA_DISK="${AUTODL_DATA_DISK:-/root/autodl-tmp}"

if [[ -n "${ENV_DIR:-}" ]]; then
  _MEDVLM_ENV_DIR="${ENV_DIR}"
elif [[ -d "${_MEDVLM_DATA_DISK}/envs/medvlm-grpo" ]]; then
  _MEDVLM_ENV_DIR="${_MEDVLM_DATA_DISK}/envs/medvlm-grpo"
else
  _MEDVLM_ENV_DIR="${_MEDVLM_PROJECT_ROOT}/.venv-autodl"
fi

if [[ -n "${CACHE_DIR:-}" ]]; then
  _MEDVLM_CACHE_DIR="${CACHE_DIR}"
elif [[ -d "${_MEDVLM_DATA_DISK}" ]]; then
  _MEDVLM_CACHE_DIR="${_MEDVLM_DATA_DISK}/cache"
else
  _MEDVLM_CACHE_DIR="${HOME}/.cache"
fi

if [[ ! -f "${_MEDVLM_ENV_DIR}/bin/activate" ]]; then
  echo "Environment not found at ${_MEDVLM_ENV_DIR}. Run bash autodl/setup.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_MEDVLM_ENV_DIR}/bin/activate"
export HF_HOME="${_MEDVLM_CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export UV_CACHE_DIR="${_MEDVLM_CACHE_DIR}/uv"
export UV_LINK_MODE=copy
export TORCH_EXTENSIONS_DIR="${_MEDVLM_CACHE_DIR}/torch_extensions"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Activated medvlm-grpo: $(python --version), env=${_MEDVLM_ENV_DIR}"
unset _MEDVLM_PROJECT_ROOT _MEDVLM_DATA_DISK _MEDVLM_ENV_DIR _MEDVLM_CACHE_DIR
