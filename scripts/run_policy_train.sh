#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_policy_train.sh RUN_NAME \
#     --dataset-repo-id=org/name [--preset=pi05_data_lerobotv3] [python args...]

RUN_NAME="${1:?Usage: bash scripts/run_policy_train.sh RUN_NAME --dataset-repo-id=org/name [options]}"
shift

export SWANLAB_API_KEY="qBi2vNBnGXoH04ErnxlUp"  # 可留空；如你不用 SwanLab 可忽略
export USR_NAME="wanghao"  # 用户名

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
evo_rl_apply_preset

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required." >&2
  exit 1
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"
CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-30000}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-${MODEL_ZOO}/physical-intelligence/evo-rl/pytorch_pi05_base_migrated}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_REPO_ID="${POLICY_REPO_ID:-unt_hub/policy_model}"
ACP_ENABLE="${ACP_ENABLE:-true}"
TAG="${TAG:-recap}"
ACP_INDICATOR_FIELD="${ACP_INDICATOR_FIELD:-complementary_info.acp_indicator_${TAG}}"
ACP_INDICATOR_DROPOUT="${ACP_INDICATOR_DROPOUT:-0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}.policy_train}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
OUTPUT_DIR="$(evo_rl_validate_output_dir "${REPO_ROOT}" "${OUTPUT_DIR}")"

DATASET_ARGS=(
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
)
POLICY_ARGS=(
  "--policy.type=${POLICY_TYPE}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.dtype=${POLICY_DTYPE}"
  "--policy.push_to_hub=${POLICY_PUSH_TO_HUB}"
  "--policy.repo_id=${POLICY_REPO_ID}"
)
if [[ -n "${POLICY_PRETRAINED_PATH}" ]]; then
  POLICY_ARGS+=("--policy.pretrained_path=${POLICY_PRETRAINED_PATH}")
fi
PI05_RENAME_MAP='{"ee_state":"observation.state","observation.ee_state":"observation.state","ee_actions":"action","observation.ee_actions":"action","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
echo "[pi05-raw32] state: ee_state[:32] -> observation.state (MIN_MAX)"
echo "[pi05-raw32] action: ee_actions[:, :32] -> action, chunk=50 (MEAN_STD)"
echo "[pi05-raw32] images: top_head/hand_left/hand_right -> observation.images.* (IDENTITY)"

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

evo_rl_configure_gpus
COMMON_ARGS=(
  "${DATASET_ARGS[@]}"
  "${POLICY_ARGS[@]}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--steps=${STEPS}"
  "--acp.enable=${ACP_ENABLE}"
  "--acp.indicator_field=${ACP_INDICATOR_FIELD}"
  "--acp.indicator_dropout_prob=${ACP_INDICATOR_DROPOUT}"
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
    -m lerobot.scripts.lerobot_train \
    "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m lerobot.scripts.lerobot_train \
    "${COMMON_ARGS[@]}"
fi
