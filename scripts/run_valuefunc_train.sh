#!/usr/bin/env bash

set -euo pipefail

# 获取输入参数
# Usage: bash scripts/run_valuefunc_train.sh <RUN_NAME> [BATCH_OR_EXTRA...]
#   - 若 $2 为纯数字：作为 batch_size（单卡 BATCH_SIZE、多卡 PER_GPU_BATCH_SIZE），其后参数为传给训练的额外参数
#   - 否则：batch 默认 32，从 $2 起全部为额外参数（与原先 OTHER_ARGS 用法一致）
# Examples:
#   bash scripts/run_valuefunc_train.sh 0422              # batch=32
#   bash scripts/run_valuefunc_train.sh 0422 64           # batch=64
#   bash scripts/run_valuefunc_train.sh 0422 16 --wandb.enable=false
#   bash scripts/run_valuefunc_train.sh 0422 --wandb.enable=false
RUN_NAME=$1
if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
  BATCH_SIZE="${2}"
  PER_GPU_BATCH_SIZE="${2}"
  shift 2
else
  BATCH_SIZE=24
  PER_GPU_BATCH_SIZE=24
  shift 1
fi
OTHER_ARGS="$*"

# ===== 必须修改 =====
export SWANLAB_API_KEY="qBi2vNBnGXoH04ErnxlUp"  # 可留空；如你不用 SwanLab 可忽略
export USR_NAME="wanghao"  # 用户名

# ===== 前置环境配置（按需修改） =====
export HF_LEROBOT_HOME="/mnt/nas/${USR_NAME}/data/lerobot_v3/"  # 数据路径
export PYTHONPATH="/mnt/nas/${USR_NAME}/openpi_05/Evo-RL"
export PYTHON="/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python"

# # export HF_ENDPOINT="https://hf-mirror.com"
# _LEROBOOT_CACHE_ROOT="/mnt/nas/.cache"
# mkdir -p "${_LEROBOOT_CACHE_ROOT}/hf_datasets" "${_LEROBOOT_CACHE_ROOT}/hf_home" "${_LEROBOOT_CACHE_ROOT}/tmp"
# export HF_DATASETS_CACHE="${_LEROBOOT_CACHE_ROOT}/hf_datasets"
# export HF_HOME="${_LEROBOOT_CACHE_ROOT}/hf_home"
# export TMPDIR="${_LEROBOOT_CACHE_ROOT}/tmp"
export MODEL_ZOO=/mnt/data/modelzoo


# ===== 训练参数（按需修改） =====
DATASET_REPO_ID="desk_basket_operation/basket_pick_plc_a02"    # "<HF_USERNAME_OR_ORG>/<DATASET_NAME>"

VALUE_TYPE="pistar06"
VALUE_DTYPE="bfloat16"
VALUE_PUSH_TO_HUB="false"
VALUE_REPO_ID="unt_hub/value_model"    # VALUE_PUSH_TO_HUB="true" 时必填
OUTPUT_DIR="outputs/value_train/${RUN_NAME}"    # "outputs/value_train/<RUN_NAME>"
JOB_NAME="${RUN_NAME}.value_train"
WANDB_ENABLE="true"

# 输出目录已存在会触发 FileExistsError，这里自动清理并重建
if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

# ===== 多卡配置 =====
USE_MULTI_GPU=1
GPU_ID_LIST="0,1,2,3,4,5,6,7"
NUM_GPUS=8

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_value_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --value.type="${VALUE_TYPE}" \
    --value.dtype="${VALUE_DTYPE}" \
    --value.push_to_hub="${VALUE_PUSH_TO_HUB}" \
    --value.repo_id="${VALUE_REPO_ID}" \
    --batch_size="${PER_GPU_BATCH_SIZE}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    ${OTHER_ARGS}
else
  "${PYTHON}" -m lerobot.scripts.lerobot_value_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --value.type="${VALUE_TYPE}" \
    --value.dtype="${VALUE_DTYPE}" \
    --value.push_to_hub="${VALUE_PUSH_TO_HUB}" \
    --value.repo_id="${VALUE_REPO_ID}" \
    --batch_size="${BATCH_SIZE}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    ${OTHER_ARGS}
fi
