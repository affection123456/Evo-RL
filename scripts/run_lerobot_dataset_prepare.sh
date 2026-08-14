#!/usr/bin/env bash
#
# 一键：本地 v2.1 -> v3.0（convert 不 merge；all 才 merge）+ augment + 可选 report。
# 源已是 v3.0 时，all/convert 走 --skip-convert 仅 merge（不改写源目录）。
#
# Usage:
#   bash scripts/run_lerobot_dataset_prepare.sh augment \
#     --dataset-repo-id=magazine_shelf_pick_20260807_0
#   bash scripts/run_lerobot_dataset_prepare.sh convert \
#     --dataset-repo-id=org/merged --sources="src_a src_b"
#
# Required:
#   --dataset-repo-id   合并/加载目标，相对 --hf-lerobot-home
#   --sources           convert/merge 必填：空格分隔的源数据集名
#
# Optional:
#   --hf-lerobot-home   默认 /mnt/nas/datasets/rldata/lerobot
#   --normalize-ee-gripper=false  关闭夹爪 [0,1000]->[0,1]（默认开启）
#   --run-dataset-report=0
#
# 路径一律：${HF_LEROBOT_HOME}/${DATASET_REPO_ID}
# merge 成功后写入 ${LOCAL_ROOT}/meta/merge_sources.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
STEP="${EVO_RL_POSITIONAL[0]:-all}"
RUN_DATASET_REPORT="${RUN_DATASET_REPORT:-1}"
PI05_ROT6D_STATS="${PI05_ROT6D_STATS:-1}"
PI05_ROT6D_DELTA="${PI05_ROT6D_DELTA:-0}"
PI05_ROT6D_STATE_DIM="${PI05_ROT6D_STATE_DIM:-32}"
PI05_ROT6D_ACTION_DIM="${PI05_ROT6D_ACTION_DIM:-32}"
NORMALIZE_EE_GRIPPER="${NORMALIZE_EE_GRIPPER:-1}"
export USR_NAME="${USR_NAME:-wanghao}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/datasets/rldata/lerobot}"
# LeRobot rejects deprecated LEROBOT_HOME; convert paths use absolute dirs below.
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
  echo "ERROR: --dataset-repo-id is required (e.g. magazine_slot_operation_200807)." >&2
  echo "  Example: bash $0 augment --dataset-repo-id=magazine_shelf_pick_20260807_0" >&2
  exit 1
fi

HF_ROOT="${HF_LEROBOT_HOME%/}"
if [[ "${DATASET_REPO_ID}" == */* ]]; then
  ORG="${DATASET_REPO_ID%%/*}"
  MERGED="${DATASET_REPO_ID##*/}"
  SOURCE_PARENT="${SOURCE_PARENT:-${ORG}}"
  NEW_REPO_ID="${NEW_REPO_ID:-${ORG}/${MERGED}_video}"
else
  ORG=""
  MERGED="${DATASET_REPO_ID}"
  SOURCE_PARENT="${SOURCE_PARENT:-.}"
  NEW_REPO_ID="${NEW_REPO_ID:-${DATASET_REPO_ID}_video}"
fi

# Absolute paths so convert/merge does not need LEROBOT_HOME.
if [[ "${SOURCE_PARENT}" == "." || -z "${SOURCE_PARENT}" ]]; then
  PARENT_DIR_ABS="${HF_ROOT}"
else
  PARENT_DIR_ABS="${HF_ROOT}/${SOURCE_PARENT}"
fi
if [[ -n "${ORG}" ]]; then
  MERGE_OUTPUT_DIR_ABS="${HF_ROOT}/${ORG}"
else
  MERGE_OUTPUT_DIR_ABS="${HF_ROOT}"
fi

LOCAL_ROOT="${HF_ROOT}/${DATASET_REPO_ID}"
MERGE_OVERWRITE="${MERGE_OVERWRITE:-0}"
CONVERT_PUSH_HUB="${CONVERT_PUSH_HUB:-0}"

_require_sources() {
  if [[ -z "${SOURCES:-}" ]]; then
    echo "ERROR: SOURCES is required for step=${STEP} (space-separated names under SOURCE_PARENT='${SOURCE_PARENT}')." >&2
    echo "  Example: SOURCES=\"magazine_slot_pick_20260807_0 magazine_slot_place_20260807_0\"" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  SOURCES_ARR=(${SOURCES})
  if [[ "${#SOURCES_ARR[@]}" -lt 1 ]]; then
    echo "ERROR: SOURCES is empty." >&2
    exit 1
  fi
}

_source_root() {
  local name="${1:?}"
  echo "${PARENT_DIR_ABS}/${name}"
}

_sources_are_v30() {
  local s root ver
  for s in "${SOURCES_ARR[@]}"; do
    root="$(_source_root "${s}")"
    if [[ ! -f "${root}/meta/info.json" ]]; then
      return 1
    fi
    ver="$("${PYTHON}" -c "import json; print(json.load(open(r'''${root}/meta/info.json''')).get('codebase_version',''))")"
    if [[ "${ver}" != "v3.0" ]]; then
      return 1
    fi
  done
  return 0
}

_write_merge_sources_json() {
  mkdir -p "${LOCAL_ROOT}/meta"
  MERGE_SOURCES_JSON_SOURCES="$(printf '%s\n' "${SOURCES_ARR[@]}")" \
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
  SOURCE_PARENT="${SOURCE_PARENT}" \
  DATASET_REPO_ID="${DATASET_REPO_ID}" \
  LOCAL_ROOT="${LOCAL_ROOT}" \
  "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

hf_home = Path(os.environ["HF_LEROBOT_HOME"]).resolve()
source_parent = os.environ.get("SOURCE_PARENT", ".")
sources = [s for s in os.environ.get("MERGE_SOURCES_JSON_SOURCES", "").splitlines() if s.strip()]
repo_id = os.environ["DATASET_REPO_ID"]
local_root = Path(os.environ["LOCAL_ROOT"])

def source_entry(name: str) -> dict:
    if source_parent in ("", "."):
        root = hf_home / name
        rel = name
    else:
        root = hf_home / source_parent / name
        rel = f"{source_parent}/{name}"
    info = {}
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "repo_id": name,
        "relative_path": rel,
        "absolute_path": str(root.resolve()),
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
    }

payload = {
    "merged_repo_id": repo_id,
    "merged_root": str(local_root.resolve()),
    "hf_lerobot_home": str(hf_home),
    "source_parent": source_parent,
    "sources": [source_entry(s) for s in sources],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
out = local_root / "meta" / "merge_sources.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote merge provenance: {out}", flush=True)
PY
}

_convert() {
  local do_merge="${1:?}"
  _require_sources

  local skip_convert="${SKIP_CONVERT:-0}"
  if [[ "${skip_convert}" != "1" ]] && _sources_are_v30; then
    skip_convert=1
    echo "Sources already v3.0 → using --skip-convert (merge-only, sources untouched)."
  fi

  local -a cmd=(
    "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/convert_local_lerobot_v21_to_v30.py"
    "--parent-dir=${PARENT_DIR_ABS}"
    "--repo-ids" "${SOURCES_ARR[@]}"
  )
  if [[ "${skip_convert}" == "1" ]]; then
    cmd+=("--skip-convert")
  fi
  if [[ "${do_merge}" == "1" ]]; then
    cmd+=("--merge-output-dir=${MERGE_OUTPUT_DIR_ABS}" "--merge-repo-id=${MERGED}")
    if [[ "${MERGE_OVERWRITE}" == "1" ]]; then
      cmd+=("--merge-overwrite")
    fi
  fi
  if [[ "${CONVERT_PUSH_HUB}" == "1" ]]; then
    cmd+=("--push-to-hub")
  fi
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "PARENT_DIR_ABS=${PARENT_DIR_ABS} MERGE_OUTPUT_DIR_ABS=${MERGE_OUTPUT_DIR_ABS}"
  echo "SOURCE_PARENT=${SOURCE_PARENT} DATASET_REPO_ID=${DATASET_REPO_ID}"
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  if [[ "${do_merge}" == "1" ]]; then
    if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
      echo "ERROR: merge finished but missing ${LOCAL_ROOT}/meta" >&2
      exit 1
    fi
    _write_merge_sources_json
  fi
}

_convert_v30_img2video() {
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "convert_v30_img2video: repo_id=${DATASET_REPO_ID} -> new_repo_id=${NEW_REPO_ID}"
  local -a cmd=(
    "${PYTHON}" -m lerobot.scripts.lerobot_edit_dataset
    "--repo_id=${DATASET_REPO_ID}"
    "--new_repo_id=${NEW_REPO_ID}"
    "--operation.type=convert_image_to_video"
    "--operation.vcodec=h264"
    "--operation.num_workers=${IMG2VIDEO_WORKERS:-16}"
  )
  if [[ -n "${IMG2VIDEO_ROOT:-}" ]]; then
    cmd+=(--root="${IMG2VIDEO_ROOT}")
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
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
    echo "ERROR: dataset root missing: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  _normalize_ee_gripper
  echo "Augment quantile stats: repo_id=${DATASET_REPO_ID} root=${LOCAL_ROOT}"
  local -a rot6d_args=()
  if [[ "${PI05_ROT6D_STATS}" == "1" ]]; then
    local _delta_tag="absolute"
    if [[ "${PI05_ROT6D_DELTA}" == "1" ]]; then
      _delta_tag="pose-delta"
    fi
    echo "  + pi05 dual-arm full32 Rot6D (${_delta_tag}, pad=${PI05_ROT6D_ACTION_DIM})"
    rot6d_args=(
      --pi05-rot6d-stats
      --rot6d-state-dim="${PI05_ROT6D_STATE_DIM}"
      --rot6d-action-dim="${PI05_ROT6D_ACTION_DIM}"
    )
    if [[ "${PI05_ROT6D_DELTA}" == "1" ]]; then
      rot6d_args+=(--pi05-rot6d-delta)
    fi
  fi
  "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py" \
    --repo-id="${DATASET_REPO_ID}" \
    --root="${LOCAL_ROOT}" \
    --overwrite \
    "${rot6d_args[@]}"
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
    _convert 1
    _augment
    ;;
  convert)
    _convert 0
    ;;
  convert_v30_img2video)
    _convert_v30_img2video
    echo "Done step=${STEP}"
    exit 0
    ;;
  augment)
    _augment
    ;;
  report)
    _report
    echo "Done step=${STEP}"
    exit 0
    ;;
  *)
    echo "Usage: bash $0 [all|convert|convert_v30_img2video|augment|report] --dataset-repo-id=name [--sources=\"a b\"]" >&2
    exit 1
    ;;
esac

echo "Done step=${STEP}"

if [[ "${RUN_DATASET_REPORT}" != "1" ]]; then
  echo "Skipping dataset report (RUN_DATASET_REPORT=0)."
  exit 0
fi

if [[ -d "${LOCAL_ROOT}/meta" ]]; then
  _report
elif [[ "${STEP}" == "all" || "${STEP}" == "augment" ]]; then
  echo "WARN: skipping dataset report (missing ${LOCAL_ROOT}/meta)." >&2
else
  echo "Skipping dataset report (step=${STEP})."
fi
