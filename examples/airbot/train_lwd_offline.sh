#!/usr/bin/env bash
set -euo pipefail
# Unified AIRBOT LWD entry: cached QAM prefixes/velocities and decoded replay images.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OPENPI_DATA_HOME="$PWD/.cache"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$PWD/.cache/divl_validation/cudnn-cu13/nvidia/cudnn/lib:$PWD/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/.cache/divl_validation/gemma-b300-deps:$PWD/.cache/airbot_lwd/gemma-runtime:$PWD/src:$PWD/packages/openpi-client/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "AIRBOT LWD: optimized QAM + decoded image cache" >&2
exec .venvs/openpi-b300/bin/python -u scripts/train_airbot_lwd.py "$@"
