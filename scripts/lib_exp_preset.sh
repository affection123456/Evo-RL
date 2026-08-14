# Shared experiment preset + CLI flag parser for policy / value train+infer scripts.
#
# PRESET → POLICY_TYPE only (dataset path is NOT bound to PRESET):
#   pi05_data_lerobotv3 — POLICY_TYPE=pi05
#   pi05_data_dmp       — POLICY_TYPE=pi05
#   pi0_dmp_data_dmp    — POLICY_TYPE=pi0_dmp
#
# All three share verified pi05_data_lerobotv3 defaults:
#   dual-arm xyz+Rot6D pose followed by the raw tail, absolute + QUANTILES.
#   The converted vector is clipped/padded to 32 dimensions.
#
# Default PRESET=pi05_data_lerobotv3 for all scripts (override via --preset=).
#
# Dataset path:
#   --hf-lerobot-home  default /mnt/nas/datasets/rldata/lerobot
#   --dataset-repo-id  required by train/infer/prepare scripts
#   --dataset-root     default ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}
#
# Examples:
#   bash scripts/run_policy_train.sh run_a --dataset-repo-id=org/name
#   bash scripts/run_valuefunc_train.sh run_b --preset=pi05_data_dmp --dataset-repo-id=dmp_data_recap/merged
#   bash scripts/run_policy_train.sh 0731_pi05_dmp \
#     --preset=pi05_data_dmp \
#     --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
#     --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot \
#     --acp-enable=false

: "${MODEL_ZOO:=/mnt/data/modelzoo}"

EVO_RL_PYTHON_ARGS=()
EVO_RL_POSITIONAL=()
_evo_rl_consumed=0
_evo_rl_cur=""
_evo_rl_nxt=""

_evo_rl_try_opt() {
  # Match --flag=value or --flag value against $_evo_rl_cur / $_evo_rl_nxt.
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
        || _evo_rl_try_opt HF_LEROBOT_HOME --hf_lerobot_home \
        || _evo_rl_try_opt DATASET_REPO_ID --dataset-repo-id \
        || _evo_rl_try_opt DATASET_REPO_ID --dataset_repo_id \
        || _evo_rl_try_opt DATASET_ROOT --dataset-root \
        || _evo_rl_try_opt DATASET_ROOT --dataset_root \
        || _evo_rl_try_opt ACP_ENABLE --acp-enable \
        || _evo_rl_try_opt ACP_ENABLE --acp_enable \
        || _evo_rl_try_opt STEPS --steps \
        || _evo_rl_try_opt BATCH_SIZE --batch-size \
        || _evo_rl_try_opt BATCH_SIZE --batch_size \
        || _evo_rl_try_opt NUM_GPUS --num-gpus \
        || _evo_rl_try_opt NUM_GPUS --num_gpus \
        || _evo_rl_try_opt USE_MULTI_GPU --use-multi-gpu \
        || _evo_rl_try_opt USE_MULTI_GPU --use_multi_gpu \
        || _evo_rl_try_opt GPU_ID_LIST --gpu-id-list \
        || _evo_rl_try_opt GPU_ID_LIST --gpu_id_list \
        || _evo_rl_try_opt PORT --port \
        || _evo_rl_try_opt PYTHON --python \
        || _evo_rl_try_opt SWANLAB_API_KEY --swanlab-api-key \
        || _evo_rl_try_opt SWANLAB_API_KEY --swanlab_api_key \
        || _evo_rl_try_opt SOURCES --sources \
        || _evo_rl_try_opt CHECKPOINT_PATH --checkpoint-path \
        || _evo_rl_try_opt CHECKPOINT_PATH --checkpoint_path \
        || _evo_rl_try_opt NUM_INFERENCE_STEPS --num-inference-steps \
        || _evo_rl_try_opt NUM_INFERENCE_STEPS --num_inference_steps \
        || _evo_rl_try_opt NORMALIZE_EE_GRIPPER --normalize-ee-gripper \
        || _evo_rl_try_opt NORMALIZE_EE_GRIPPER --normalize_ee_gripper \
        || _evo_rl_try_opt RUN_DATASET_REPORT --run-dataset-report
      then
        shift "${_evo_rl_consumed}"
        continue
      fi
      # Unknown --* goes to the python/lerobot command (e.g. --policy.xxx, --wandb.enable=false).
      EVO_RL_PYTHON_ARGS+=("$1")
      shift
      continue
    fi
    EVO_RL_POSITIONAL+=("$1")
    shift
  done
}

evo_rl_apply_preset() {
  PRESET="${PRESET:-pi05_data_lerobotv3}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/datasets/rldata/lerobot}"
  : "${USR_NAME:=wanghao}"
  : "${MODEL_ZOO:=/mnt/data/modelzoo}"

  case "${PRESET}" in
    pi0_dmp_data_dmp)
      POLICY_TYPE="pi0_dmp"
      ;;
    pi05_data_lerobotv3)
      POLICY_TYPE="pi05"
      ;;
    pi05_data_dmp)
      POLICY_TYPE="pi05"
      ;;
    *)
      echo "ERROR: unknown PRESET='${PRESET}' (use pi05_data_lerobotv3 | pi05_data_dmp | pi0_dmp_data_dmp)" >&2
      exit 1
      ;;
  esac

  if [[ -n "${DATASET_REPO_ID:-}" ]]; then
    DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME%/}/${DATASET_REPO_ID}}"
  fi

  POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-${MODEL_ZOO}/physical-intelligence/evo-rl/pytorch_pi05_base_migrated}"

  echo "[preset] PRESET=${PRESET} POLICY_TYPE=${POLICY_TYPE}"
  echo "[preset] HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "[preset] EE layout: dual-arm full32 Rot6D"
  if [[ -n "${DATASET_REPO_ID:-}" ]]; then
    echo "[preset] DATASET_REPO_ID=${DATASET_REPO_ID}"
    echo "[preset] DATASET_ROOT=${DATASET_ROOT}"
  else
    echo "[preset] DATASET_REPO_ID=<unset>"
  fi
}
