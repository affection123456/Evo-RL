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

# ===== 推理参数（按需修改） =====
DATASET_REPO_ID="desk_basket_operation/basket_pick_plc_a02"    # "<HF_USERNAME_OR_ORG>/<DATASET_NAME>"
CHECKPOINT_PATH="outputs/value_train/${RUN_NAME}"

RUNTIME_DEVICE="cuda"
RUNTIME_BATCH_SIZE=64
ACP_ENABLE="true"
ACP_N_STEP=50
ACP_POSITIVE_RATIO=0.3
TAG="recap"
ACP_VALUE_FIELD="complementary_info.value_${TAG}"
ACP_ADV_FIELD="complementary_info.advantage_${TAG}"
ACP_IND_FIELD="complementary_info.acp_indicator_${TAG}"
OUTPUT_DIR="outputs/value_infer/${RUN_NAME}"
JOB_NAME="${RUN_NAME}.infer"

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
    -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --inference.checkpoint_path="${CHECKPOINT_PATH}" \
    --runtime.device="${RUNTIME_DEVICE}" \
    --runtime.batch_size="${RUNTIME_BATCH_SIZE}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.n_step="${ACP_N_STEP}" \
    --acp.positive_ratio="${ACP_POSITIVE_RATIO}" \
    --acp.value_field="${ACP_VALUE_FIELD}" \
    --acp.advantage_field="${ACP_ADV_FIELD}" \
    --acp.indicator_field="${ACP_IND_FIELD}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    ${OTHER_ARGS}
else
  "${PYTHON}" -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --inference.checkpoint_path="${CHECKPOINT_PATH}" \
    --runtime.device="${RUNTIME_DEVICE}" \
    --runtime.batch_size="${RUNTIME_BATCH_SIZE}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.n_step="${ACP_N_STEP}" \
    --acp.positive_ratio="${ACP_POSITIVE_RATIO}" \
    --acp.value_field="${ACP_VALUE_FIELD}" \
    --acp.advantage_field="${ACP_ADV_FIELD}" \
    --acp.indicator_field="${ACP_IND_FIELD}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    ${OTHER_ARGS}
fi
