#!/usr/bin/env bash
#
# pi0_dmp 专用：已是 v3.0 的 DMP 源数据 → merge → absolute Rot6D quantile augment (pad32) → report。
# 与 pi05_data_lerobotv3 对齐：QUANTILES + pad32 + 默认不用 delta（PI0_DMP_ROT6D_DELTA=1 可开）。
#
# Usage:
#   bash scripts/run_lerobot_dataset_prepare_pi0_dmp.sh augment \
#     --dataset-repo-id=dmp_data_recap/merged
#   bash scripts/run_lerobot_dataset_prepare_pi0_dmp.sh merge \
#     --dataset-repo-id=dmp_data_recap/merged --sources="src_a src_b"
#
# Required:
#   --dataset-repo-id   合并输出，相对 --hf-lerobot-home，如 dmp_data_recap/unt_merged_xxx
#   --sources           merge 必填：空格分隔的源数据集名
#
# Optional:
#   --hf-lerobot-home   默认 /mnt/nas/datasets/rldata/lerobot
#   --normalize-ee-gripper=false  关闭夹爪 [0,1000]->[0,1]（默认开启）
#   --run-dataset-report=0
#
# 路径一律：${HF_LEROBOT_HOME}/${DATASET_REPO_ID}
# basket / v2.1→v3.0 请用 scripts/run_lerobot_dataset_prepare.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
STEP="${EVO_RL_POSITIONAL[0]:-all}"
RUN_DATASET_REPORT="${RUN_DATASET_REPORT:-1}"
NORMALIZE_EE_GRIPPER="${NORMALIZE_EE_GRIPPER:-1}"
export USR_NAME="${USR_NAME:-wanghao}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/datasets/rldata/lerobot}"
unset LEROBOT_HOME

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"

_LEROBOOT_CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
mkdir -p "${_LEROBOOT_CACHE_ROOT}/hf_datasets" "${_LEROBOOT_CACHE_ROOT}/hf_home" "${_LEROBOOT_CACHE_ROOT}/tmp"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${_LEROBOOT_CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${_LEROBOOT_CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${_LEROBOOT_CACHE_ROOT}/tmp}"

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required (e.g. dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot)." >&2
  echo "  Example: bash $0 augment --dataset-repo-id=dmp_data_recap/merged" >&2
  exit 1
fi
if [[ "${DATASET_REPO_ID}" != */* ]]; then
  echo "ERROR: DATASET_REPO_ID must be 'parent/name' (got '${DATASET_REPO_ID}')." >&2
  exit 1
fi

SOURCE_PARENT="${SOURCE_PARENT:-dmp_data}"
MERGE_PARENT="${DATASET_REPO_ID%%/*}"
MERGED="${DATASET_REPO_ID##*/}"
HF_ROOT="${HF_LEROBOT_HOME%/}"
PARENT_DIR_ABS="${HF_ROOT}/${SOURCE_PARENT}"
MERGE_OUTPUT_DIR_ABS="${HF_ROOT}/${MERGE_PARENT}"
LOCAL_ROOT="${HF_ROOT}/${DATASET_REPO_ID}"

MERGE_OVERWRITE="${MERGE_OVERWRITE:-0}"
ROT6D_STATE_DIM="${PI0_DMP_ROT6D_STATE_DIM:-32}"
ROT6D_ACTION_DIM="${PI0_DMP_ROT6D_ACTION_DIM:-32}"
PI0_DMP_ROT6D_DELTA="${PI0_DMP_ROT6D_DELTA:-0}"

_require_sources() {
  if [[ -z "${SOURCES:-}" ]]; then
    echo "ERROR: SOURCES is required for step=${STEP} (space-separated names under ${SOURCE_PARENT})." >&2
    echo "  Example: SOURCES=\"unt_0511_shelf_pik_DMP_lerobot unt_0616_Mz_right_pik_DMP_lerobot\"" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  SOURCES_ARR=(${SOURCES})
  if [[ "${#SOURCES_ARR[@]}" -lt 1 ]]; then
    echo "ERROR: SOURCES is empty." >&2
    exit 1
  fi
}

_required_keys=(
  ee_state
  ee_actions
  ref_ee_state
  ref_ee_actions
  top_head
  hand_right
  ref_top_head
  ref_hand_right
)

_validate_dataset() {
  local root="${1:?}"
  local label="${2:-dataset}"
  if [[ ! -d "${root}/meta" ]]; then
    echo "ERROR: ${label} missing meta/: ${root}" >&2
    exit 1
  fi
  "${PYTHON}" - <<'PY' "${root}" "${label}" "${_required_keys[@]}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
required = sys.argv[3:]
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"ERROR: {label} missing {info_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
ver = info.get("codebase_version")
if ver != "v3.0":
    raise SystemExit(f"ERROR: {label} codebase_version={ver!r}, expected 'v3.0' ({root})")
features = info.get("features") or {}
missing = [k for k in required if k not in features]
if missing:
    raise SystemExit(f"ERROR: {label} missing features {missing} ({root})")

ee_state = features["ee_state"].get("shape") or []
ee_actions = features["ee_actions"].get("shape") or []
if not ee_state or int(ee_state[-1]) < 14:
    raise SystemExit(f"ERROR: {label} ee_state shape invalid: {ee_state}")
if len(ee_actions) < 1 or int(ee_actions[-1]) < 14:
    raise SystemExit(f"ERROR: {label} ee_actions shape invalid: {ee_actions}")

stats_path = root / "meta" / "stats.json"
n_eps = info.get("total_episodes")
n_frames = info.get("total_frames")
print(
    f"[ok] {label}: v3.0 eps={n_eps} frames={n_frames} "
    f"ee_state={list(ee_state)} ee_actions={list(ee_actions)} "
    f"stats={'yes' if stats_path.is_file() else 'no'} root={root}",
    flush=True,
)
PY
}

_merge() {
  _require_sources
  echo "Validate merge sources under ${PARENT_DIR_ABS}"
  local s
  for s in "${SOURCES_ARR[@]}"; do
    _validate_dataset "${PARENT_DIR_ABS}/${s}" "source:${s}"
  done

  local -a cmd=(
    "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/convert_local_lerobot_v21_to_v30.py"
    "--parent-dir=${PARENT_DIR_ABS}"
    "--repo-ids" "${SOURCES_ARR[@]}"
    "--skip-convert"
    "--merge-output-dir=${MERGE_OUTPUT_DIR_ABS}"
    "--merge-repo-id=${MERGED}"
  )
  if [[ "${MERGE_OVERWRITE}" == "1" ]]; then
    cmd+=("--merge-overwrite")
  fi
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "PARENT_DIR_ABS=${PARENT_DIR_ABS} MERGE_OUTPUT_DIR_ABS=${MERGE_OUTPUT_DIR_ABS}"
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  _validate_dataset "${LOCAL_ROOT}" "merged:${DATASET_REPO_ID}"
}

_normalize_ee_gripper() {
  case "${NORMALIZE_EE_GRIPPER}" in
    0|false|FALSE|no|NO)
      echo "Skip EE gripper normalize (NORMALIZE_EE_GRIPPER=${NORMALIZE_EE_GRIPPER})"
      return 0
      ;;
  esac
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing for gripper normalize: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  echo "Normalize EE gripper [0,1000]->[0,1]: ${LOCAL_ROOT}"
  "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/normalize_ee_gripper_scale.py" \
    --root="${LOCAL_ROOT}"
}

_augment() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: merged dataset root missing: ${LOCAL_ROOT}" >&2
    echo "HINT: run merge first, or check DATASET_REPO_ID." >&2
    exit 1
  fi
  _validate_dataset "${LOCAL_ROOT}" "augment-input:${DATASET_REPO_ID}"
  _normalize_ee_gripper
  local _delta_tag="absolute"
  local -a _delta_args=()
  if [[ "${PI0_DMP_ROT6D_DELTA}" == "1" ]]; then
    _delta_tag="pose-delta"
    _delta_args=(--pi0-dmp-rot6d-delta)
  fi
  echo "Augment quantile stats (pi0_dmp dual-arm full32 Rot6D/${_delta_tag}, pad=${ROT6D_STATE_DIM}): repo_id=${DATASET_REPO_ID} root=${LOCAL_ROOT}"
  "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py" \
    --repo-id="${DATASET_REPO_ID}" \
    --root="${LOCAL_ROOT}" \
    --overwrite \
    --pi0-dmp-rot6d-stats \
    --rot6d-state-dim="${ROT6D_STATE_DIM}" \
    --rot6d-action-dim="${ROT6D_ACTION_DIM}" \
    "${_delta_args[@]}"

  "${PYTHON}" - <<'PY' "${LOCAL_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stats = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
needed = [
    "observation.state",
    "observation.reference.state",
    "action",
    "observation.ref_actions",
]
missing = [k for k in needed if k not in stats]
if missing:
    raise SystemExit(f"ERROR: rot6d stats missing keys after augment: {missing}")
for key in needed:
    shape = (stats[key].get("mean") or stats[key].get("min") or [])
    if hasattr(shape, "__len__") and len(shape) == 0:
        raise SystemExit(f"ERROR: empty stats for {key}")
print(f"[ok] rot6d stats keys present: {needed}", flush=True)
PY
}

_report() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing for report: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  echo "Running dataset report: ${DATASET_REPO_ID} (root=${HF_LEROBOT_HOME%/})"
  "${PYTHON}" -m lerobot.scripts.lerobot_dataset_report \
    --dataset "${DATASET_REPO_ID}" \
    --root "${HF_LEROBOT_HOME%/}"
}

case "${STEP}" in
  all)
    _merge
    _augment
    ;;
  merge)
    _merge
    ;;
  augment)
    _augment
    ;;
  report)
    _report
    echo "Done (step=${STEP})."
    exit 0
    ;;
  *)
    echo "Usage: DATASET_REPO_ID=parent/name SOURCES=\"a b\" $0 [all|merge|augment|report]" >&2
    exit 1
    ;;
esac

echo "Done (step=${STEP})."

if [[ "${RUN_DATASET_REPORT}" == "1" ]]; then
  if [[ -d "${LOCAL_ROOT}/meta" ]]; then
    _report
  else
    echo "WARN: skipping dataset report (missing ${LOCAL_ROOT}/meta)." >&2
  fi
else
  echo "Skipping dataset report (RUN_DATASET_REPORT=0)."
fi
