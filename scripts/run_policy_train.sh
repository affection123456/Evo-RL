#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_policy_train.sh <RUN_NAME> [--preset=...] [--dataset-repo-id=...] [python args...]
#
# Examples:
#   bash scripts/run_policy_train.sh 0813_pi05_magazine_shelf_right \
#     --dataset-repo-id=magazine_shelf_pick_20260807_0 --acp-enable=false
#   bash scripts/run_policy_train.sh 0731_pi05_dmp \
#     --preset=pi05_data_dmp \
#     --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
#     --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot \
#     --acp-enable=false

RUN_NAME="${1:?Usage: bash scripts/run_policy_train.sh <RUN_NAME> [--preset=...] [--dataset-repo-id=...] ...}"
shift 1

# ===== 必须修改 =====
export SWANLAB_API_KEY="qBi2vNBnGXoH04ErnxlUp"  # 可留空；如你不用 SwanLab 可忽略
export USR_NAME="wanghao"  # 用户名

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
evo_rl_apply_preset

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required (e.g. --dataset-repo-id=org/name)." >&2
  exit 1
fi
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME%/}/${DATASET_REPO_ID}}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"
_LEROBOOT_CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${_LEROBOOT_CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${_LEROBOOT_CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-/tmp/evo-rl_${USR_NAME}}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"
export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"

POLICY_DEVICE="cuda"
POLICY_DTYPE="bfloat16"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${POLICY_TYPE}" == "pi0_dmp" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=8
  fi
fi
STEPS="${STEPS:-30000}"
ACP_ENABLE="${ACP_ENABLE:-true}"
TAG="${TAG:-recap}"
ACP_INDICATOR_FIELD="${ACP_INDICATOR_FIELD:-complementary_info.acp_indicator_${TAG}}"
ACP_INDICATOR_DROPOUT="${ACP_INDICATOR_DROPOUT:-0.3}"
OUTPUT_DIR="outputs/train/${RUN_NAME}"
JOB_NAME="${RUN_NAME}.policy_train"
WANDB_ENABLE="true"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_REPO_ID="unt_hub/policy_model"

POLICY_EXTRA_ARGS=()
if [[ "${POLICY_TYPE}" == "pi0_dmp" ]]; then
  POLICY_EXTRA_ARGS=(
    --policy.use_rot6d="${EE_USE_ROT6D}"
    --policy.ee_arm_mode="${EE_ARM_MODE}"
    --policy.ee_gripper_dims="${EE_GRIPPER_DIMS}"
    --policy.rot6d_delta_action=false
    --policy.max_state_dim=32
    --policy.max_action_dim=32
    --policy.ref_state_key="observation.reference.state"
    --policy.ref_action_key="observation.ref_actions"
    --policy.tokenizer_max_length=200
    --rename_map='{"ee_state":"observation.state","ee_actions":"action","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","ref_ee_state":"observation.reference.state","ref_ee_actions":"observation.ref_actions","ref_top_head":"observation.images.ref_top_head","ref_hand_right":"observation.images.ref_hand_right"}'
  )
elif [[ "${POLICY_TYPE}" == "pi05" ]]; then
  # Shared EE contract: right xyz+rotation+gripper_first(1) by default.
  # use_rot6d=true => 10 physical dims; false => 8; model head stays pad32.
  POLICY_EXTRA_ARGS=(
    --policy.use_rot6d="${EE_USE_ROT6D}"
    --policy.ee_arm_mode="${EE_ARM_MODE}"
    --policy.ee_gripper_dims="${EE_GRIPPER_DIMS}"
    --policy.rot6d_delta_action=false
    --policy.max_state_dim=32
    --policy.max_action_dim=32
    --policy.ee_state_key="observation.ee_state"
    --policy.ee_action_key="observation.ee_actions"
    --rename_map='{"ee_state":"observation.ee_state","ee_actions":"observation.ee_actions","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
  )
fi

DATASET_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
)

POLICY_PRETRAINED_ARGS=()
if [[ -n "${POLICY_PRETRAINED_PATH}" ]]; then
  POLICY_PRETRAINED_ARGS+=(--policy.pretrained_path="${POLICY_PRETRAINED_PATH}")
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

_DETECTED_NUM_GPUS=1
if command -v nvidia-smi >/dev/null 2>&1; then
  _DETECTED_NUM_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
  if [[ "${_DETECTED_NUM_GPUS}" -lt 1 ]]; then
    _DETECTED_NUM_GPUS=1
  fi
fi
NUM_GPUS="${NUM_GPUS:-${_DETECTED_NUM_GPUS}}"
if [[ -z "${GPU_ID_LIST:-}" ]]; then
  if [[ "${NUM_GPUS}" -le 1 ]]; then
    GPU_ID_LIST="0"
  else
    GPU_ID_LIST="$(seq -s, 0 $((NUM_GPUS - 1)))"
  fi
fi
if [[ -z "${USE_MULTI_GPU:-}" ]]; then
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    USE_MULTI_GPU=1
  else
    USE_MULTI_GPU=0
  fi
fi
PER_GPU_BATCH_SIZE="${BATCH_SIZE}"

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${DATASET_ARGS[@]}" \
    --policy.type="${POLICY_TYPE}" \
    "${POLICY_PRETRAINED_ARGS[@]}" \
    --policy.device="${POLICY_DEVICE}" \
    --policy.dtype="${POLICY_DTYPE}" \
    --batch_size="${PER_GPU_BATCH_SIZE}" \
    --steps="${STEPS}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.indicator_field="${ACP_INDICATOR_FIELD}" \
    --acp.indicator_dropout_prob="${ACP_INDICATOR_DROPOUT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --policy.push_to_hub="${POLICY_PUSH_TO_HUB}" \
    --policy.repo_id="${POLICY_REPO_ID}" \
    "${POLICY_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
else
  "${PYTHON}" -m lerobot.scripts.lerobot_train \
    "${DATASET_ARGS[@]}" \
    --policy.type="${POLICY_TYPE}" \
    "${POLICY_PRETRAINED_ARGS[@]}" \
    --policy.device="${POLICY_DEVICE}" \
    --policy.dtype="${POLICY_DTYPE}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.indicator_field="${ACP_INDICATOR_FIELD}" \
    --acp.indicator_dropout_prob="${ACP_INDICATOR_DROPOUT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --policy.push_to_hub="${POLICY_PUSH_TO_HUB}" \
    --policy.repo_id="${POLICY_REPO_ID}" \
    "${POLICY_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
fi
