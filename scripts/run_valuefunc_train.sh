#!/usr/bin/env bash

set -euo pipefail

# 获取输入参数
# Usage: bash scripts/run_valuefunc_train.sh <RUN_NAME> [--batch-size=16] [--dataset-repo-id=...] ...
#   bash scripts/run_valuefunc_train.sh 0422 --dataset-repo-id=org/name
#   bash scripts/run_valuefunc_train.sh 0422 --batch-size=16 --wandb.enable=false --dataset-repo-id=org/name
RUN_NAME="${1:?Usage: bash scripts/run_valuefunc_train.sh <RUN_NAME> [--batch-size=...] [--dataset-repo-id=...] ...}"
shift 1

# ===== 必须修改 =====
export SWANLAB_API_KEY="qBi2vNBnGXoH04ErnxlUp"  # 可留空；如你不用 SwanLab 可忽略
export USR_NAME="wanghao"  # 用户名

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
BATCH_SIZE="${BATCH_SIZE:-16}"
PER_GPU_BATCH_SIZE="${BATCH_SIZE}"
evo_rl_apply_preset

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required (e.g. --dataset-repo-id=org/name)." >&2
  exit 1
fi
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME%/}/${DATASET_REPO_ID}}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"

# HuggingFace datasets 会在读取 parquet 时生成 cache；不要落到已满的根分区或 /mnt/data。
_LEROBOOT_CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${_LEROBOOT_CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${_LEROBOOT_CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-/tmp/evo-rl_${USR_NAME}}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"
export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"

VALUE_TYPE="pistar06"
VALUE_DTYPE="bfloat16"
VALUE_PUSH_TO_HUB="false"
VALUE_REPO_ID="unt_hub/value_model"    # VALUE_PUSH_TO_HUB="true" 时必填
OUTPUT_DIR="outputs/value_train/${RUN_NAME}"    # "outputs/value_train/<RUN_NAME>"
JOB_NAME="${RUN_NAME}.value_train"
WANDB_ENABLE="true"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

VALUE_EXTRA_ARGS=()
if [[ "${POLICY_TYPE}" == "pi0_dmp" ]]; then
  VALUE_EXTRA_ARGS=(
    --value.enable_stage_heads=true
    --value.include_ref_state_in_prompt=true
    --value.max_state_dim=32
    --value.state_feature="observation.state"
    --value.ref_state_feature="observation.reference.state"
    --value.camera_features="['observation.images.top_head','observation.images.hand_right']"
    --value.reference_camera_features="['observation.images.ref_top_head','observation.images.ref_hand_right']"
    --value.num_stage_classes=8
    --rename_map='{"ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","ref_ee_state":"observation.reference.state","ref_top_head":"observation.images.ref_top_head","ref_hand_right":"observation.images.ref_hand_right"}'
  )
elif [[ "${POLICY_TYPE}" == "pi05" ]]; then
  # EE quat → observation.state → Rot6D pad32；rename 兼容 DMP bare keys 与 basket observation.ee_*
  VALUE_EXTRA_ARGS=(
    --value.max_state_dim=32
    --value.state_feature="observation.state"
    --rename_map='{"ee_state":"observation.state","observation.ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
  )
  # DMP meta uses bare video keys (top_head/...), so auto camera detect from
  # observation.images.* is empty unless we set camera_features explicitly.
  if [[ "${PRESET}" == "pi05_data_dmp" ]]; then
    VALUE_EXTRA_ARGS+=(
      --value.camera_features="['observation.images.top_head','observation.images.hand_right']"
    )
  fi
fi

DATASET_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
)

# 输出目录已存在会触发 FileExistsError，这里自动清理并重建
if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

# ===== 多卡配置 =====
USE_MULTI_GPU="${USE_MULTI_GPU:-1}"
GPU_ID_LIST="${GPU_ID_LIST:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_value_train \
    "${DATASET_ARGS[@]}" \
    --value.type="${VALUE_TYPE}" \
    --value.dtype="${VALUE_DTYPE}" \
    --value.push_to_hub="${VALUE_PUSH_TO_HUB}" \
    --value.repo_id="${VALUE_REPO_ID}" \
    --batch_size="${PER_GPU_BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint="${SAVE_CHECKPOINT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    "${VALUE_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
else
  "${PYTHON}" -m lerobot.scripts.lerobot_value_train \
    "${DATASET_ARGS[@]}" \
    --value.type="${VALUE_TYPE}" \
    --value.dtype="${VALUE_DTYPE}" \
    --value.push_to_hub="${VALUE_PUSH_TO_HUB}" \
    --value.repo_id="${VALUE_REPO_ID}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint="${SAVE_CHECKPOINT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    "${VALUE_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
fi
