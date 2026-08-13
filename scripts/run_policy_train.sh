#!/usr/bin/env bash

set -euo pipefail

# 获取输入参数
RUN_NAME=$1
OTHER_ARGS=${2:-}

# ===== 必须修改 =====
export SWANLAB_API_KEY="qBi2vNBnGXoH04ErnxlUp"  # 可留空；如你不用 SwanLab 可忽略
export USR_NAME="wanghao"  # 用户名

# ===== 前置环境配置（按需修改） =====
export HF_LEROBOT_HOME="/mnt/nas/${USR_NAME}/data/lerobot_v3/"  # 数据路径
export PYTHONPATH="/mnt/nas/${USR_NAME}/openpi_05/Evo-RL"
export PYTHON="/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python"
export MODEL_ZOO=/mnt/data/modelzoo

# ===== 训练参数（按需修改） =====
DATASET_REPO_ID="desk_basket_operation/basket_pick_plc_a02"    # "<HF_USERNAME_OR_ORG>/<DATASET_NAME>"

# lerobot_train 只接受 pi0 / pi05 等策略名；pistar06 是价值模型，见 run_valuefunc_train.sh
POLICY_TYPE="pi05"
POLICY_PRETRAINED_PATH="${MODEL_ZOO}/physical-intelligence/evo-rl/pytorch_pi05_base_migrated"
POLICY_DEVICE="cuda"
POLICY_DTYPE="bfloat16"
BATCH_SIZE=8
STEPS=30000
ACP_ENABLE="true"
TAG="recap"
ACP_INDICATOR_FIELD="complementary_info.acp_indicator_${TAG}"
ACP_INDICATOR_DROPOUT=0.3
OUTPUT_DIR="outputs/train/${RUN_NAME}"
JOB_NAME="${RUN_NAME}.policy_train"
WANDB_ENABLE="true"
# 离线训练默认不 push，避免训练结束因 huggingface.co 不可达而报错退出。
# 需要在线上传时可在命令前覆盖：
#   POLICY_PUSH_TO_HUB=true bash scripts/run_policy_train.sh <RUN_NAME>
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_REPO_ID="unt_hub/policy_model"    # POLICY_PUSH_TO_HUB="true" 时必填

# 输出目录已存在会触发 FileExistsError，这里自动清理并重建
if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output dir exists, removing: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

# ===== 多卡配置 =====
USE_MULTI_GPU=1
GPU_ID_LIST="0,1,2,3,4,5,6,7"
NUM_GPUS=8
PER_GPU_BATCH_SIZE="${BATCH_SIZE}"

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --policy.type="${POLICY_TYPE}" \
    --policy.pretrained_path="${POLICY_PRETRAINED_PATH}" \
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
    ${OTHER_ARGS}
else
  "${PYTHON}" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --policy.type="${POLICY_TYPE}" \
    --policy.pretrained_path="${POLICY_PRETRAINED_PATH}" \
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
    ${OTHER_ARGS}
fi
