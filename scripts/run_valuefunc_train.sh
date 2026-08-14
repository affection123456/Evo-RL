#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_valuefunc_train.sh RUN_NAME \
#     --dataset-repo-id=org/name [--batch-size=16] [python args...]

RUN_NAME="${1:?Usage: bash scripts/run_valuefunc_train.sh RUN_NAME --dataset-repo-id=org/name [options]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
evo_rl_apply_preset

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required." >&2
  exit 1
fi

USR_NAME="${USR_NAME:-wanghao}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"
CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VALUE_TYPE="${VALUE_TYPE:-pistar06}"
VALUE_DTYPE="${VALUE_DTYPE:-bfloat16}"
VALUE_PUSH_TO_HUB="${VALUE_PUSH_TO_HUB:-false}"
VALUE_REPO_ID="${VALUE_REPO_ID:-unt_hub/value_model}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/value_train/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}.value_train}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
PI05_RENAME_MAP='{"ee_state":"observation.state","observation.ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
OUTPUT_DIR="$(evo_rl_validate_output_dir "${REPO_ROOT}" "${OUTPUT_DIR}")"
VALUE_STEPS_ARGS=()
if [[ -n "${STEPS:-}" ]]; then
  VALUE_STEPS_ARGS+=("--steps=${STEPS}")
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

evo_rl_configure_gpus
COMMON_ARGS=(
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--value.type=${VALUE_TYPE}"
  "--value.dtype=${VALUE_DTYPE}"
  "--value.push_to_hub=${VALUE_PUSH_TO_HUB}"
  "--value.repo_id=${VALUE_REPO_ID}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "${VALUE_STEPS_ARGS[@]}"
  "--log_freq=${LOG_FREQ}"
  "--save_checkpoint=${SAVE_CHECKPOINT}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--rename_map=${PI05_RENAME_MAP}"
  "${EVO_RL_PYTHON_ARGS[@]}"
)

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    "--num_processes=${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_value_train \
    "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m lerobot.scripts.lerobot_value_train \
    "${COMMON_ARGS[@]}"
fi
