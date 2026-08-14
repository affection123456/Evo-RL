#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_valuefunc_infer.sh RUN_NAME \
#     --dataset-repo-id=org/name [--checkpoint-path=...] [python args...]

RUN_NAME="${1:?Usage: bash scripts/run_valuefunc_infer.sh RUN_NAME --dataset-repo-id=org/name [options]}"
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

RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
RUNTIME_BATCH_SIZE="${RUNTIME_BATCH_SIZE:-${BATCH_SIZE:-64}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACP_ENABLE="${ACP_ENABLE:-true}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/value_train/${RUN_NAME}}"
ACP_N_STEP="${ACP_N_STEP:-50}"
ACP_POSITIVE_RATIO="${ACP_POSITIVE_RATIO:-0.3}"
TAG="${TAG:-recap}"
ACP_VALUE_FIELD="${ACP_VALUE_FIELD:-complementary_info.value_${TAG}}"
ACP_ADV_FIELD="${ACP_ADV_FIELD:-complementary_info.advantage_${TAG}}"
ACP_IND_FIELD="${ACP_IND_FIELD:-complementary_info.acp_indicator_${TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/value_infer/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}.infer}"
PI05_RENAME_MAP='{"ee_state":"observation.state","observation.ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
OUTPUT_DIR="$(evo_rl_validate_output_dir "${REPO_ROOT}" "${OUTPUT_DIR}")"

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

evo_rl_configure_gpus
COMMON_ARGS=(
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--inference.checkpoint_path=${CHECKPOINT_PATH}"
  "--runtime.device=${RUNTIME_DEVICE}"
  "--runtime.batch_size=${RUNTIME_BATCH_SIZE}"
  "--runtime.num_workers=${NUM_WORKERS}"
  "--acp.enable=${ACP_ENABLE}"
  "--acp.n_step=${ACP_N_STEP}"
  "--acp.positive_ratio=${ACP_POSITIVE_RATIO}"
  "--acp.value_field=${ACP_VALUE_FIELD}"
  "--acp.advantage_field=${ACP_ADV_FIELD}"
  "--acp.indicator_field=${ACP_IND_FIELD}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--rename_map=${PI05_RENAME_MAP}"
  "${EVO_RL_PYTHON_ARGS[@]}"
)

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    "--num_processes=${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_value_infer \
    "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m lerobot.scripts.lerobot_value_infer \
    "${COMMON_ARGS[@]}"
fi
