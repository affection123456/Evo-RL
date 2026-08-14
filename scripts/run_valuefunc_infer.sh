#!/usr/bin/env bash

set -euo pipefail

# Usage: bash scripts/run_valuefunc_infer.sh <RUN_NAME> [--dataset-repo-id=...] [--preset=...] ...
RUN_NAME="${1:?Usage: bash scripts/run_valuefunc_infer.sh <RUN_NAME> [--dataset-repo-id=...] ...}"
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
_DEFAULT_TMP_ROOT="/tmp/evo-rl_${USR_NAME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${_LEROBOOT_CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${_LEROBOOT_CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${_DEFAULT_TMP_ROOT}}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"
export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"

RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
RUNTIME_BATCH_SIZE="${RUNTIME_BATCH_SIZE:-64}"
RUNTIME_NUM_WORKERS="${NUM_WORKERS:-${RUNTIME_NUM_WORKERS:-4}}"
ACP_ENABLE="${ACP_ENABLE:-true}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/value_train/${RUN_NAME}}"
ACP_N_STEP=50
ACP_POSITIVE_RATIO=0.3
TAG="recap"
ACP_VALUE_FIELD="complementary_info.value_${TAG}"
ACP_ADV_FIELD="complementary_info.advantage_${TAG}"
ACP_IND_FIELD="complementary_info.acp_indicator_${TAG}"
ACP_REF_STAGE_FIELD="complementary_info.ref_stage_${TAG}"
ACP_REF_STAGE_INDEX_FIELD="complementary_info.ref_stage_index_${TAG}"
ACP_REF_STAGE_PROGRESS_FIELD="complementary_info.ref_stage_progress_${TAG}"
ACP_REF_STAGE_VALUE_FIELD="complementary_info.ref_stage_value_${TAG}"
OUTPUT_DIR="outputs/value_infer/${RUN_NAME}"
JOB_NAME="${RUN_NAME}.infer"

INFER_EXTRA_ARGS=()
if [[ "${POLICY_TYPE}" == "pi0_dmp" ]]; then
  INFER_EXTRA_ARGS=(
    --acp.emit_ref_stage=true
    --acp.ref_stage_field="${ACP_REF_STAGE_FIELD}"
    --acp.ref_stage_index_field="${ACP_REF_STAGE_INDEX_FIELD}"
    --acp.ref_stage_progress_field="${ACP_REF_STAGE_PROGRESS_FIELD}"
    --acp.ref_stage_value_field="${ACP_REF_STAGE_VALUE_FIELD}"
    --rename_map='{"ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","ref_ee_state":"observation.reference.state","ref_top_head":"observation.images.ref_top_head","ref_hand_right":"observation.images.ref_hand_right"}'
  )
elif [[ "${POLICY_TYPE}" == "pi05" ]]; then
  INFER_EXTRA_ARGS=(
    --rename_map='{"ee_state":"observation.state","observation.ee_state":"observation.state","top_head":"observation.images.top_head","hand_right":"observation.images.hand_right","hand_left":"observation.images.hand_left"}'
  )
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

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}" "${PYTHON}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_value_infer \
    "${DATASET_ARGS[@]}" \
    --inference.checkpoint_path="${CHECKPOINT_PATH}" \
    --runtime.device="${RUNTIME_DEVICE}" \
    --runtime.batch_size="${RUNTIME_BATCH_SIZE}" \
    --runtime.num_workers="${RUNTIME_NUM_WORKERS}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.n_step="${ACP_N_STEP}" \
    --acp.positive_ratio="${ACP_POSITIVE_RATIO}" \
    --acp.value_field="${ACP_VALUE_FIELD}" \
    --acp.advantage_field="${ACP_ADV_FIELD}" \
    --acp.indicator_field="${ACP_IND_FIELD}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    "${INFER_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
else
  "${PYTHON}" -m lerobot.scripts.lerobot_value_infer \
    "${DATASET_ARGS[@]}" \
    --inference.checkpoint_path="${CHECKPOINT_PATH}" \
    --runtime.device="${RUNTIME_DEVICE}" \
    --runtime.batch_size="${RUNTIME_BATCH_SIZE}" \
    --runtime.num_workers="${RUNTIME_NUM_WORKERS}" \
    --acp.enable="${ACP_ENABLE}" \
    --acp.n_step="${ACP_N_STEP}" \
    --acp.positive_ratio="${ACP_POSITIVE_RATIO}" \
    --acp.value_field="${ACP_VALUE_FIELD}" \
    --acp.advantage_field="${ACP_ADV_FIELD}" \
    --acp.indicator_field="${ACP_IND_FIELD}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    "${INFER_EXTRA_ARGS[@]}" \
    "${EVO_RL_PYTHON_ARGS[@]}"
fi
