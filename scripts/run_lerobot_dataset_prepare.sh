#!/usr/bin/env bash
#
# 一键：本地 v2.1 -> v3.0（convert 不 merge；all 才 merge）+ augment + 可选 report。
#
# Usage:
#   bash scripts/run_lerobot_dataset_prepare.sh              # all: convert+merge -> augment -> report
#   bash scripts/run_lerobot_dataset_prepare.sh convert      # 仅 v21->v30，不 merge
#   bash scripts/run_lerobot_dataset_prepare.sh convert_v30_img2video   # v3 image->video(h264)，默认源为合并后的 REPO_ID
#   bash scripts/run_lerobot_dataset_prepare.sh augment
#   bash scripts/run_lerobot_dataset_prepare.sh report
#
# 只改下面「==== 数据集 ====」块即可。image->video 源默认同合并后的 REPO_ID；否则运行前 export REPO_ID=org/name。
# 输出默认 NEW_REPO_ID=org/合并名_video（可 export 覆盖）。可选 IMG2VIDEO_ROOT（--root）、IMG2VIDEO_WORKERS。
#
#   RUN_DATASET_REPORT=0 bash ...   # 跳过 all/convert/augment 末尾的 report
#
set -euo pipefail

STEP="${1:-all}"
RUN_DATASET_REPORT="${RUN_DATASET_REPORT:-1}"

# ===== 与 run_valuefunc_train.sh 一致（按需修改） =====
export USR_NAME="wanghao"
export HF_LEROBOT_HOME="/mnt/nas/${USR_NAME}/data/lerobot_v3/"
unset LEROBOT_HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHON="/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python"

_LEROBOOT_CACHE_ROOT="/mnt/nas/.cache"
mkdir -p "${_LEROBOOT_CACHE_ROOT}/hf_datasets" "${_LEROBOOT_CACHE_ROOT}/hf_home" "${_LEROBOOT_CACHE_ROOT}/tmp"
export HF_DATASETS_CACHE="${_LEROBOOT_CACHE_ROOT}/hf_datasets"
export HF_HOME="${_LEROBOOT_CACHE_ROOT}/hf_home"
export TMPDIR="${_LEROBOOT_CACHE_ROOT}/tmp"

# ===== 数据集（只改这里）=====
ORG="desk_basket_pick"
SOURCES=(
  "basket_pick_0422_lerobot_poor"
  "basket_pick_0422_lerobot"
)
MERGED="basket_pick_0422_s_140_f_9"

MERGE_OVERWRITE=0
CONVERT_PUSH_HUB=0

REPO_ID="${REPO_ID:-${ORG}/${MERGED}}"
LOCAL_ROOT="${HF_LEROBOT_HOME%/}/${REPO_ID}"
NEW_REPO_ID="${NEW_REPO_ID:-${ORG}/${MERGED}_video}"

_convert() {
  local do_merge="${1:?}"
  local -a cmd=(
    "${PYTHON}" "${REPO_ROOT}/scripts/convert_local_lerobot_v21_to_v30.py"
    "--parent-dir=lerobot/${ORG}"
    "--repo-ids" "${SOURCES[@]}"
  )
  if [[ "${do_merge}" == "1" ]]; then
    cmd+=("--merge-output-dir=lerobot_v3/${ORG}" "--merge-repo-id=${MERGED}")
    if [[ "${MERGE_OVERWRITE}" == "1" ]]; then
      cmd+=("--merge-overwrite")
    fi
  fi
  if [[ "${CONVERT_PUSH_HUB}" == "1" ]]; then
    cmd+=("--push-to-hub")
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

_convert_v30_img2video() {
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "convert_v30_img2video: repo_id=${REPO_ID} -> new_repo_id=${NEW_REPO_ID}"
  local -a cmd=(
    "${PYTHON}" -m lerobot.scripts.lerobot_edit_dataset
    "--repo_id=${REPO_ID}"
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

_augment() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  echo "Augment quantile stats: repo_id=${REPO_ID} root=${LOCAL_ROOT}"
  "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py" \
    --repo-id="${REPO_ID}" \
    --root="${LOCAL_ROOT}" \
    --overwrite
}

_report() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing for report: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  echo "Running dataset report: ${REPO_ID} (root=${HF_LEROBOT_HOME%/})"
  "${PYTHON}" -m lerobot.scripts.lerobot_dataset_report \
    --dataset "${REPO_ID}" \
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
    echo "Done (step=${STEP})."
    exit 0
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
    echo "Usage: $0 [all|convert|convert_v30_img2video|augment|report]" >&2
    exit 1
    ;;
esac

echo "Done (step=${STEP})."

if [[ "${RUN_DATASET_REPORT}" == "1" ]]; then
  if [[ -d "${LOCAL_ROOT}/meta" ]]; then
    _report
  elif [[ "${STEP}" == "all" ]] || [[ "${STEP}" == "augment" ]]; then
    echo "WARN: skipping dataset report (missing ${LOCAL_ROOT}/meta)." >&2
  else
    echo "Skipping dataset report (step=${STEP})."
  fi
else
  echo "Skipping dataset report (RUN_DATASET_REPORT=0)."
fi
