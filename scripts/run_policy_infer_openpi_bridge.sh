#!/usr/bin/env bash

set -euo pipefail

# Serve a trained policy over the OpenPI msgpack WebSocket bridge.
#
# Usage:
#   bash scripts/run_policy_infer_openpi_bridge.sh <RUN_NAME> [--port=8003] [--preset=...]
#   bash scripts/run_policy_infer_openpi_bridge.sh 0731_pi05_dmp --port=8003 --preset=pi05_data_dmp

RUN_NAME="${1:?Usage: bash scripts/run_policy_infer_openpi_bridge.sh <RUN_NAME> [--port=8003] [--preset=...]}"
shift 1

export USR_NAME="${USR_NAME:-wanghao}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
if [[ -z "${PORT:-}" && -n "${EVO_RL_POSITIONAL[0]:-}" && "${EVO_RL_POSITIONAL[0]}" =~ ^[0-9]+$ ]]; then
  PORT="${EVO_RL_POSITIONAL[0]}"
fi
PORT="${PORT:-8003}"
evo_rl_apply_preset

export MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:/mnt/nas/${USR_NAME}/openpi_05/openpi/src:/mnt/nas/${USR_NAME}/openpi_05/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/train/${RUN_NAME}/checkpoints/last}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
PI0_DMP_KV_CACHE="${PI0_DMP_KV_CACHE:-false}"

if [[ "${POLICY_TYPE}" == "pi0_dmp" ]]; then
  RENAME_MAP='{"observation/state":"observation.state","reference/state":"observation.reference.state","ref_actions":"observation.ref_actions","observation/image":"observation.images.top_head","observation/right_wrist_image":"observation.images.hand_right","reference/image":"observation.images.ref_top_head","reference/right_wrist_image":"observation.images.ref_hand_right"}'
elif [[ "${POLICY_TYPE}" == "pi05" ]]; then
  RENAME_MAP='{"observation/state":"observation.ee_state","observation/image":"observation.images.top_head","observation/right_wrist_image":"observation.images.hand_right","observation/left_wrist_image":"observation.images.hand_left"}'
else
  echo "ERROR: unsupported POLICY_TYPE=${POLICY_TYPE}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${PI0_DMP_KV_CACHE}" == "true" ]]; then
  if [[ "${POLICY_TYPE}" != "pi0_dmp" ]]; then
    echo "ERROR: PI0_DMP_KV_CACHE=true is only valid for PRESET=pi0_dmp_data_dmp" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--pi0_dmp_kv_cache=true)
fi

echo "[infer-bridge] PRESET=${PRESET} POLICY_TYPE=${POLICY_TYPE}"
echo "[infer-bridge] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[infer-bridge] port=${PORT} num_inference_steps=${NUM_INFERENCE_STEPS}"

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint path does not exist: ${CHECKPOINT_PATH}" >&2
  echo "  Expected e.g. outputs/train/${RUN_NAME}/checkpoints/last (with pretrained_model/)." >&2
  echo "  Current run may only have SwanLab logs, or checkpoints were deleted/overwritten." >&2
  exit 1
fi

exec "${PYTHON}" -m lerobot.scripts.lerobot_policy_infer_openpi_bridge \
  --policy.path="${CHECKPOINT_PATH}" \
  --policy.device=cuda \
  --policy.num_inference_steps="${NUM_INFERENCE_STEPS}" \
  --port="${PORT}" \
  --rename_map="${RENAME_MAP}" \
  "${EXTRA_ARGS[@]}" \
  "${EVO_RL_PYTHON_ARGS[@]}"
