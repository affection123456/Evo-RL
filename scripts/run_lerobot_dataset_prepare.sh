#!/usr/bin/env bash
#
# Prepare a local LeRobot dataset:
#   all: convert sources, merge them, augment quantile stats, optionally report
#   convert: convert sources without merging
#   convert_v30_img2video: convert an existing v3 image dataset to video
#   augment/report: operate on the requested dataset directly
#
# Example:
#   bash scripts/run_lerobot_dataset_prepare.sh all \
#     --dataset-repo-id=org/merged --sources="source_a source_b"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_exp_preset.sh
source "${SCRIPT_DIR}/lib_exp_preset.sh"
evo_rl_parse_script_args "$@"
evo_rl_apply_preset

STEP="${EVO_RL_POSITIONAL[0]:-all}"
RUN_DATASET_REPORT="${RUN_DATASET_REPORT:-1}"
USR_NAME="${USR_NAME:-wanghao}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/mnt/data/miniconda3/envs/evo-rl_${USR_NAME}/bin/python}"

CACHE_ROOT="${EVO_RL_CACHE_ROOT:-/mnt/nas/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TMPDIR}"

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "ERROR: --dataset-repo-id is required." >&2
  exit 1
fi

HF_ROOT="${HF_LEROBOT_HOME%/}"
LOCAL_ROOT="${DATASET_ROOT:-${HF_ROOT}/${DATASET_REPO_ID}}"
if [[ "${DATASET_REPO_ID}" == */* ]]; then
  ORG="${DATASET_REPO_ID%%/*}"
  MERGED="${DATASET_REPO_ID#*/}"
  SOURCE_PARENT="${SOURCE_PARENT:-${ORG}}"
  MERGE_OUTPUT_DIR="${HF_ROOT}/${ORG}"
  NEW_REPO_ID="${NEW_REPO_ID:-${ORG}/${MERGED}_video}"
else
  ORG=""
  MERGED="${DATASET_REPO_ID}"
  SOURCE_PARENT="${SOURCE_PARENT:-.}"
  MERGE_OUTPUT_DIR="${HF_ROOT}"
  NEW_REPO_ID="${NEW_REPO_ID:-${DATASET_REPO_ID}_video}"
fi
if [[ "${SOURCE_PARENT}" == "." || -z "${SOURCE_PARENT}" ]]; then
  SOURCE_DIR="${HF_ROOT}"
else
  SOURCE_DIR="${HF_ROOT}/${SOURCE_PARENT}"
fi

_require_sources() {
  if [[ -z "${SOURCES:-}" ]]; then
    echo "ERROR: --sources is required for step=${STEP} (space-separated dataset names)." >&2
    exit 1
  fi
  read -r -a SOURCES_ARR <<< "${SOURCES}"
  if [[ "${#SOURCES_ARR[@]}" -eq 0 ]]; then
    echo "ERROR: --sources is empty." >&2
    exit 1
  fi
}

_classify_source_versions() {
  local source version
  local v21_count=0
  local v30_count=0
  for source in "${SOURCES_ARR[@]}"; do
    if [[ ! -f "${SOURCE_DIR}/${source}/meta/info.json" ]]; then
      echo "ERROR: missing source metadata: ${SOURCE_DIR}/${source}/meta/info.json" >&2
      return 1
    fi
    version="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("codebase_version", ""))' \
      "${SOURCE_DIR}/${source}/meta/info.json")"
    case "${version}" in
      v2.1) ((v21_count += 1)) ;;
      v3.0) ((v30_count += 1)) ;;
      *)
        echo "ERROR: unsupported codebase_version='${version}' for source '${source}'." >&2
        return 1
        ;;
    esac
  done
  if [[ "${v21_count}" -gt 0 && "${v30_count}" -gt 0 ]]; then
    echo "ERROR: mixed v2.1/v3.0 sources are unsafe; convert v2.1 sources first, then merge only v3.0 sources." >&2
    return 1
  fi
  if [[ "${v30_count}" -gt 0 ]]; then
    printf 'v3.0\n'
  else
    printf 'v2.1\n'
  fi
}

_convert() {
  local merge="$1"
  _require_sources
  local source_version
  source_version="$(_classify_source_versions)"
  local skip_convert="${SKIP_CONVERT:-0}"
  if [[ "${source_version}" == "v3.0" ]]; then
    skip_convert=1
    echo "Sources are already v3.0; using merge-only mode."
  fi
  local -a cmd=(
    "${PYTHON}" "${REPO_ROOT}/scripts/convert_local_lerobot_v21_to_v30.py"
    "--parent-dir=${SOURCE_DIR}"
    "--repo-ids" "${SOURCES_ARR[@]}"
  )
  if [[ "${merge}" == "1" ]]; then
    cmd+=("--merge-output-dir=${MERGE_OUTPUT_DIR}" "--merge-repo-id=${MERGED}")
    if [[ "${MERGE_OVERWRITE:-0}" == "1" ]]; then
      cmd+=("--merge-overwrite")
    fi
  fi
  if [[ "${skip_convert}" == "1" ]]; then
    cmd+=("--skip-convert")
  fi
  if [[ "${CONVERT_PUSH_HUB:-0}" == "1" ]]; then
    cmd+=("--push-to-hub")
  fi
  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

_convert_v30_img2video() {
  local -a cmd=(
    "${PYTHON}" -m lerobot.scripts.lerobot_edit_dataset
    "--repo_id=${DATASET_REPO_ID}"
    "--new_repo_id=${NEW_REPO_ID}"
    "--operation.type=convert_image_to_video"
    "--operation.vcodec=h264"
    "--operation.num_workers=${IMG2VIDEO_WORKERS:-16}"
    "--root=${LOCAL_ROOT}"
  )
  "${cmd[@]}"
}

_augment() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  "${PYTHON}" "${REPO_ROOT}/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py" \
    "--repo-id=${DATASET_REPO_ID}" \
    "--root=${LOCAL_ROOT}" \
    --overwrite
}

_report() {
  if [[ ! -d "${LOCAL_ROOT}/meta" ]]; then
    echo "ERROR: dataset root missing for report: ${LOCAL_ROOT}" >&2
    exit 1
  fi
  "${PYTHON}" -m lerobot.scripts.lerobot_dataset_report \
    --dataset "${LOCAL_ROOT}"
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
    ;;
  augment)
    _augment
    ;;
  report)
    _report
    ;;
  *)
    echo "Usage: bash $0 [all|convert|convert_v30_img2video|augment|report] --dataset-repo-id=org/name [--sources=\"a b\"]" >&2
    exit 1
    ;;
esac

if [[ "${RUN_DATASET_REPORT}" == "1" && "${STEP}" != "report" && -d "${LOCAL_ROOT}/meta" ]]; then
  _report
fi
