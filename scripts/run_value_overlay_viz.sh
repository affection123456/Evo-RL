#!/usr/bin/env bash
# Render top_head camera videos with value curve overlay (Evo-RL value_infer_viz style).
#
# Default preset: 0515_1 OOD on basket_place_a02_04 (10 success + 10 failure episodes).
#
# Examples:
#   ./scripts/run_value_overlay_viz.sh
#   EPISODES=0,116 OUTPUT_DIR=outputs/value_infer/0515_1_ood/viz_manual ./scripts/run_value_overlay_viz.sh
#   EPISODES=all OUTPUT_DIR=outputs/value_infer/0515_1_ood/viz_all OVERWRITE=1 ./scripts/run_value_overlay_viz.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
export USR_NAME="${USR_NAME:-wanghao}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/${USR_NAME}/data/lerobot_v3}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"
_LEROBOOT_CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
_DEFAULT_TMP_ROOT="/tmp/evo-rl_${USR_NAME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${_LEROBOOT_CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${_LEROBOOT_CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${_DEFAULT_TMP_ROOT}}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"

DATASET_REPO_ID="${DATASET_REPO_ID:-desk_basket_place/basket_place_a02_04_s_116_f_24}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/nas/${USR_NAME}/data/lerobot_v3/desk_basket_place/basket_place_a02_04_s_116_f_24}"
VALUE_TAG="${VALUE_TAG:-0515_1_ood}"
VALUE_FIELD="${VALUE_FIELD:-complementary_info.value_${VALUE_TAG}}"
ADVANTAGE_FIELD="${ADVANTAGE_FIELD:-complementary_info.advantage_${VALUE_TAG}}"
INDICATOR_FIELD="${INDICATOR_FIELD:-complementary_info.acp_indicator_${VALUE_TAG}}"
VIDEO_KEY="${VIDEO_KEY:-observation.images.top_head}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
VCODEC="${VCODEC:-h264}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-1}"
EPISODES="${EPISODES:-0-9,116-125}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/value_infer/${VALUE_TAG}_a02_04_s10_f10_pyav/value/viz}"
OVERWRITE="${OVERWRITE:-0}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

echo "=========================================="
echo "Evo-RL value overlay visualization"
echo "=========================================="
echo "Dataset:      ${DATASET_ROOT}"
echo "Repo id:      ${DATASET_REPO_ID}"
echo "Episodes:     ${EPISODES}"
echo "Value field:  ${VALUE_FIELD}"
echo "Video key:    ${VIDEO_KEY}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Video backend:${VIDEO_BACKEND}"
echo "=========================================="

"${PYTHON}" -m lerobot.datasets.render_value_overlay_on_video \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${DATASET_REPO_ID}" \
  --episodes "${EPISODES}" \
  --value-field "${VALUE_FIELD}" \
  --advantage-field "${ADVANTAGE_FIELD}" \
  --indicator-field "${INDICATOR_FIELD}" \
  --video-key "${VIDEO_KEY}" \
  --output-dir "${OUTPUT_DIR}" \
  --video-backend "${VIDEO_BACKEND}" \
  --vcodec "${VCODEC}" \
  --smooth-window "${SMOOTH_WINDOW}" \
  "${OVERWRITE_ARGS[@]}" \
  "${EVO_RL_PYTHON_ARGS[@]}"
