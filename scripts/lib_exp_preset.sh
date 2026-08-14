#!/usr/bin/env bash

# Shared PI05 experiment presets and safe CLI parsing for the shell entrypoints.
# Unknown options and their values are forwarded as individual Python arguments.
# Use `--` before passthrough values that themselves begin with `--`.

EVO_RL_PYTHON_ARGS=()
EVO_RL_POSITIONAL=()
_evo_rl_consumed=0
_evo_rl_cur=""
_evo_rl_nxt=""

_evo_rl_try_opt() {
  local var_name="$1"
  local flag="$2"
  _evo_rl_consumed=0
  if [[ "${_evo_rl_cur}" == "${flag}="* ]]; then
    printf -v "${var_name}" '%s' "${_evo_rl_cur#*=}"
    _evo_rl_consumed=1
    return 0
  fi
  if [[ "${_evo_rl_cur}" == "${flag}" ]]; then
    if [[ -z "${_evo_rl_nxt}" || "${_evo_rl_nxt}" == --* ]]; then
      echo "ERROR: ${flag} requires a value" >&2
      exit 1
    fi
    printf -v "${var_name}" '%s' "${_evo_rl_nxt}"
    _evo_rl_consumed=2
    return 0
  fi
  return 1
}

evo_rl_parse_script_args() {
  EVO_RL_PYTHON_ARGS=()
  EVO_RL_POSITIONAL=()
  while [[ $# -gt 0 ]]; do
    _evo_rl_cur="$1"
    _evo_rl_nxt="${2:-}"
    if [[ "${_evo_rl_cur}" == "--" ]]; then
      shift
      EVO_RL_PYTHON_ARGS+=("$@")
      break
    fi
    if [[ "${_evo_rl_cur}" == --* ]]; then
      if _evo_rl_try_opt PRESET --preset \
        || _evo_rl_try_opt HF_LEROBOT_HOME --hf-lerobot-home \
        || _evo_rl_try_opt DATASET_REPO_ID --dataset-repo-id \
        || _evo_rl_try_opt DATASET_ROOT --dataset-root \
        || _evo_rl_try_opt BATCH_SIZE --batch-size \
        || _evo_rl_try_opt NUM_WORKERS --num-workers \
        || _evo_rl_try_opt RUNTIME_BATCH_SIZE --runtime-batch-size \
        || _evo_rl_try_opt NUM_GPUS --num-gpus \
        || _evo_rl_try_opt USE_MULTI_GPU --use-multi-gpu \
        || _evo_rl_try_opt GPU_ID_LIST --gpu-id-list \
        || _evo_rl_try_opt PYTHON --python \
        || _evo_rl_try_opt EVO_RL_CACHE_ROOT --cache-root \
        || _evo_rl_try_opt HF_DATASETS_CACHE --hf-datasets-cache \
        || _evo_rl_try_opt HF_HOME --hf-home \
        || _evo_rl_try_opt TMPDIR --tmpdir \
        || _evo_rl_try_opt MODEL_ZOO --model-zoo \
        || _evo_rl_try_opt ACP_ENABLE --acp-enable \
        || _evo_rl_try_opt STEPS --steps \
        || _evo_rl_try_opt SOURCES --sources \
        || _evo_rl_try_opt SOURCE_PARENT --source-parent \
        || _evo_rl_try_opt CHECKPOINT_PATH --checkpoint-path \
        || _evo_rl_try_opt RUN_DATASET_REPORT --run-dataset-report
      then
        shift "${_evo_rl_consumed}"
        continue
      fi
      EVO_RL_PYTHON_ARGS+=("$1")
      if [[ -n "${2:-}" && "${2}" != --* ]]; then
        EVO_RL_PYTHON_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      continue
    fi
    EVO_RL_POSITIONAL+=("$1")
    shift
  done
}

evo_rl_apply_preset() {
  PRESET="${PRESET:-pi05_data_lerobotv3}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/datasets/rldata/lerobot}"
  unset LEROBOT_HOME
  USR_NAME="${USR_NAME:-wanghao}"
  MODEL_ZOO="${MODEL_ZOO:-/mnt/data/modelzoo}"

  case "${PRESET}" in
    pi05_data_lerobotv3|pi05_data_dmp)
      POLICY_TYPE="pi05"
      ;;
    *)
      echo "ERROR: unknown --preset '${PRESET}' (use pi05_data_lerobotv3 or pi05_data_dmp)" >&2
      return 1
      ;;
  esac

  if [[ -n "${DATASET_REPO_ID:-}" ]]; then
    DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME%/}/${DATASET_REPO_ID}}"
  fi
  POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-${MODEL_ZOO}/physical-intelligence/evo-rl/pytorch_pi05_base_migrated}"

  echo "[preset] PRESET=${PRESET} POLICY_TYPE=${POLICY_TYPE}"
  echo "[preset] HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "[preset] DATASET_REPO_ID=${DATASET_REPO_ID:-<unset>}"
  if [[ -n "${DATASET_ROOT:-}" ]]; then
    echo "[preset] DATASET_ROOT=${DATASET_ROOT}"
  fi
}
