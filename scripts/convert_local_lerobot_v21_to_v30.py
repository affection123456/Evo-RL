#!/usr/bin/env python3
"""Convert local LeRobot datasets from v2.1 to v3.0 (Evo-RL / current LeRobot format).

Your folders already report ``codebase_version: "v2.1"`` in ``meta/info.json``. The
``lerobot==0.1.0`` pip version is unrelated; conversion is driven by ``codebase_version``.

This script wraps the upstream converter:

  ``src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py``

which rewrites layout (chunked parquet, ``meta/episodes`` as parquet, etc.) and sets
``codebase_version`` to ``v3.0``.

By default the upstream converter is run with ``--keep-original``: v3.0 is written beside the
source as ``<parent>/<repo_id>_v30`` (using the dataset folder basename), and the original v2.1
folder is left unchanged. Use ``--in-place-convert`` to use the legacy behavior (source moved to
``_old``, v3.0 takes the original name).

If ``--merge-output-dir`` is provided, each converted v3 dataset is copied under that directory
first (as a new parent dir for merge inputs), and merge is also executed there.

Example::

python scripts/convert_local_lerobot_v21_to_v30.py \
        --parent-dir lerobot/desk_basket_pick \
        --repo-ids basket_pick_0416_vla_lerobot basket_pick_0416_vla_lerobot_poor \
        --merge-output-dir lerobot_v3/desk_basket_pick \
        --merge-repo-id basket_pick_0416_vla_s_5_f_5

Relative paths resolve under ``LEROBOT_HOME`` (env ``LEROBOT_HOME``, else parent of
``HF_LEROBOT_HOME`` if set, else ``/mnt/nas/wanghao/data``).

Merged data is written to ``{LEROBOT_HOME}/lerobot_v3/desk_basket_pick/basket_pick_0416_vla_s_5_f_5_v30`` (parent may exist).
Converted v3 sources used for merge are placed under
``{LEROBOT_HOME}/lerobot_v3/desk_basket_pick/<repo_id>_v30`` by default
(e.g. ``{LEROBOT_HOME}/lerobot_v3/desk_basket_pick/basket_pick_0416_vla_lerobot_v30``).

Optional merge requires the same feature schema in all sources. Legacy ``rating`` in
``episodes.jsonl`` is merged into v3 episode metadata by the upstream converter.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# Root for resolving relative --parent-dir / --merge-output-dir / etc.
# Prefer explicit LEROBOT_HOME; otherwise strip trailing slash from HF_LEROBOT_HOME
# (e.g. .../lerobot_v3 -> parent data root); else default for backwards compatibility.
_hf = os.environ.get("HF_LEROBOT_HOME", "").strip().rstrip("/")
_default_lerobot_home = os.path.dirname(_hf) if _hf else "/mnt/nas/wanghao/data"
LEROBOT_HOME = os.environ.get("LEROBOT_HOME", _default_lerobot_home)
DEFAULT_V3_PARENT = "lerobot_v3"

# Rating -> episode_success mapping macros (edit these to fit your dataset conventions).
RATING_SOURCE_COLUMN = "rating"
SUCCESS_TARGET_COLUMN = "episode_success"
RATING_TO_SUCCESS_LABEL = {
    "excellent": "success",
    "acceptable": "success",
    "poor": "failure",
}
UNKNOWN_SUCCESS_LABEL = "unlabeled"

# Repack mapping (source key -> canonical key). Applied after conversion.
# This mirrors pi0-style preprocessing at the dataset level so training scripts
# can consume canonical observation/action keys directly.
DATA_REPACK_MAP = {
    "state": "observation.state",
    "actions": "action",
    "top_head": "observation.images.top_head",
    "hand_left": "observation.images.hand_left",
    "hand_right": "observation.images.hand_right",
    "ee_state": "observation.ee_state",
    "ee_actions": "observation.ee_actions",
    "camera_pose": "observation.camera_pose",
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_data_path(path: Path) -> Path:
    """Resolve absolute paths directly, and resolve relative paths under LEROBOT_HOME."""
    if path.is_absolute():
        return path.resolve()
    return (Path(LEROBOT_HOME) / path).resolve()


def _load_codebase_version(dataset_dir: Path) -> str:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    return str(info.get("codebase_version", "unknown"))


def _map_rating_to_success(value: object) -> str:
    if value is None:
        return UNKNOWN_SUCCESS_LABEL
    key = str(value).strip().lower()
    if not key:
        return UNKNOWN_SUCCESS_LABEL
    return RATING_TO_SUCCESS_LABEL.get(key, UNKNOWN_SUCCESS_LABEL)


def _write_episode_success_from_rating(dataset_root: Path) -> None:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        return

    updated_files = 0
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        if RATING_SOURCE_COLUMN not in df.columns:
            continue
        df[SUCCESS_TARGET_COLUMN] = df[RATING_SOURCE_COLUMN].map(_map_rating_to_success)
        df.to_parquet(parquet_file, index=False)
        updated_files += 1

    if updated_files > 0:
        print(
            f"Wrote {SUCCESS_TARGET_COLUMN} from {RATING_SOURCE_COLUMN} in {updated_files} episode parquet files.",
            flush=True,
        )


def _rename_columns_with_conflict_drop(df: pd.DataFrame, rename_map: dict[str, str]) -> tuple[pd.DataFrame, bool]:
    rename_ops: dict[str, str] = {}
    drop_cols: list[str] = []
    for source_key, target_key in rename_map.items():
        if source_key not in df.columns:
            continue
        if target_key in df.columns:
            drop_cols.append(source_key)
        else:
            rename_ops[source_key] = target_key

    changed = bool(rename_ops or drop_cols)
    if rename_ops:
        df = df.rename(columns=rename_ops)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, changed


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _repack_rename_media_dirs(dataset_root: Path, effective_map: dict[str, str]) -> int:
    """Rename ``videos/<src>`` / ``images/<src>`` directories to match repacked feature keys (``<dst>``)."""
    renamed = 0
    for media_root_name in ("videos", "images"):
        media_root = dataset_root / media_root_name
        if not media_root.is_dir():
            continue
        for src, dst in effective_map.items():
            src_dir = media_root / src
            dst_dir = media_root / dst
            if not src_dir.is_dir():
                continue
            if dst_dir.exists():
                if src_dir.resolve() == dst_dir.resolve():
                    continue
                print(
                    f"Warning: skip {media_root_name} rename {src!r} -> {dst!r}: destination exists ({dst_dir}).",
                    flush=True,
                )
                continue
            src_dir.rename(dst_dir)
            renamed += 1
    return renamed


def _repack_converted_dataset_keys(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    features: dict[str, dict] = dict(info.get("features", {}))
    if not features:
        return

    effective_map = {
        src: dst
        for src, dst in DATA_REPACK_MAP.items()
        if src in features and src != dst
    }
    if not effective_map:
        return

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    data_updates = 0
    for parquet_file in data_files:
        df = pd.read_parquet(parquet_file)
        df, changed = _rename_columns_with_conflict_drop(df, effective_map)
        if changed:
            df.to_parquet(parquet_file, index=False)
            data_updates += 1

    episodes_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    episode_updates = 0
    prefixed_map: dict[str, str] = {}
    for src, dst in effective_map.items():
        prefixed_map[f"stats/{src}/"] = f"stats/{dst}/"
        prefixed_map[f"videos/{src}/"] = f"videos/{dst}/"

    for parquet_file in episodes_files:
        df = pd.read_parquet(parquet_file)
        rename_ops: dict[str, str] = {}
        for col in df.columns:
            for src_prefix, dst_prefix in prefixed_map.items():
                if col.startswith(src_prefix):
                    rename_ops[col] = f"{dst_prefix}{col[len(src_prefix):]}"
                    break
        if rename_ops:
            df = df.rename(columns=rename_ops)
            df.to_parquet(parquet_file, index=False)
            episode_updates += 1

    for src, dst in effective_map.items():
        feature_payload = features.get(src)
        if feature_payload is None:
            continue
        if dst not in features:
            features[dst] = feature_payload
        del features[src]
    info["features"] = features
    _write_json(info_path, info)

    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.is_file():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        stats_changed = False
        for src, dst in effective_map.items():
            if src in stats:
                if dst not in stats:
                    stats[dst] = stats[src]
                del stats[src]
                stats_changed = True
        if stats_changed:
            _write_json(stats_path, stats)

    # Parquet + info now reference canonical keys (e.g. videos/observation.images.top_head/...), but the
    # converter still writes files under videos/top_head/ etc. Rename on disk so LeRobotDataset can find
    # them offline (otherwise _check_cached_episodes_sufficient fails and init tries the Hub).
    media_renames = _repack_rename_media_dirs(dataset_root, effective_map)

    extras = f", media_dirs={media_renames}" if media_renames else ""
    print(
        "Repacked converted keys "
        f"({len(effective_map)} mappings, data_files={data_updates}, episode_files={episode_updates}{extras}) "
        f"for {dataset_root}.",
        flush=True,
    )


def _postprocess_converted_dataset(dataset_root: Path) -> None:
    _write_episode_success_from_rating(dataset_root)
    _repack_converted_dataset_keys(dataset_root)


def _run_v21_to_v30_direct(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    force: bool,
    push_to_hub: bool,
) -> None:
    """
    Convert v2.1 -> v3.0 from source_root/repo_id directly into target_root.

    This bypasses the upstream CLI default "<source>_v30 beside source" behavior so the converted
    dataset is created directly under lerobot_v3.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        convert_data,
        convert_episodes_metadata,
        convert_info,
        convert_tasks,
        convert_videos,
        validate_local_dataset_version,
    )

    source_ds = (source_root / repo_id).resolve()
    target_ds = target_root.resolve()
    if target_ds.exists():
        if force:
            shutil.rmtree(target_ds)
        else:
            print(
                f"Converted v3 dataset already exists: {target_ds}\n"
                "Use --force-conversion to replace it or choose a different --converted-output-dir.",
                file=sys.stderr,
            )
            sys.exit(1)
    target_ds.parent.mkdir(parents=True, exist_ok=True)

    validate_local_dataset_version(source_ds)
    print(f"Converting (direct): {source_ds} -> {target_ds}", flush=True)
    convert_info(source_ds, target_ds, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    convert_tasks(source_ds, target_ds)
    episodes_metadata = convert_data(source_ds, target_ds, DEFAULT_DATA_FILE_SIZE_IN_MB)
    episodes_videos_metadata = convert_videos(source_ds, target_ds, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    convert_episodes_metadata(source_ds, target_ds, episodes_metadata, episodes_videos_metadata)
    _postprocess_converted_dataset(target_ds)
    print(f"Converted v3.0 dataset written to {target_ds}.", flush=True)

    if push_to_hub:
        LeRobotDataset(repo_id, root=str(target_ds)).push_to_hub()


def _v30_sibling_path(parent_dir: Path, repo_id: str) -> Path:
    """Path to ``<dataset_folder>_v30`` next to ``parent_dir / repo_id``."""
    src = (parent_dir / repo_id).resolve()
    return src.parent / f"{src.name}_v30"


def _expected_v30_path(
    parent_dir: Path,
    repo_id: str,
    converted_output_dir: Path,
) -> Path:
    """
    Default v3 destination under converted_output_dir.

    If parent_dir is under ``{LEROBOT_HOME}/lerobot``, keep its subfolder structure there
    (e.g. ``lerobot/desk_basket_pick`` -> ``lerobot_v3/desk_basket_pick``).
    """
    home = Path(LEROBOT_HOME).resolve()
    parent_rel = Path()
    legacy_root = home / "lerobot"
    if parent_dir.is_relative_to(legacy_root):
        parent_rel = parent_dir.relative_to(legacy_root)

    repo_path = Path(repo_id)
    v30_repo_name = f"{repo_path.name}_v30"
    return (converted_output_dir / parent_rel / repo_path.parent / v30_repo_name).resolve()


def _relocate_converted_dataset(
    source_v30_dir: Path,
    target_v30_dir: Path,
    overwrite: bool,
) -> Path:
    src = source_v30_dir.resolve()
    dst = target_v30_dir.resolve()
    if src == dst:
        return dst

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            print(
                f"Converted v3 dataset already exists: {dst}\n"
                "Use --force-conversion to replace it or choose a different --converted-output-dir.",
                file=sys.stderr,
            )
            sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"Relocated converted dataset: {src} -> {dst}", flush=True)
    return dst


def _merge_local(
    dataset_roots: list[Path],
    output_dir: Path,
    merge_repo_id: str,
) -> None:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets: list[LeRobotDataset] = []
    for root in dataset_roots:
        root = root.resolve()
        # Use the on-disk folder name as repo_id so local opens stay local; synthetic ids trigger Hub lookups.
        datasets.append(LeRobotDataset(root.name, root=str(root)))

    merge_datasets(datasets, output_repo_id=merge_repo_id, output_dir=output_dir)
    print(f"Merged dataset written to {output_dir} (repo_id={merge_repo_id}).", flush=True)


def _sync_merge_sources(
    dataset_roots: list[Path],
    merge_parent: Path,
    overwrite: bool,
) -> list[Path]:
    synced_roots: list[Path] = []
    for src in dataset_roots:
        src_resolved = src.resolve()
        dst = (merge_parent / src_resolved.name).resolve()
        if dst == src_resolved:
            synced_roots.append(dst)
            continue
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                print(
                    f"Converted dataset target already exists: {dst}\n"
                    "Use --merge-overwrite to replace it, or choose a different --merge-output-dir.",
                    file=sys.stderr,
                )
                sys.exit(1)
        shutil.copytree(src_resolved, dst)
        print(f"Synced merge source: {src_resolved} -> {dst}", flush=True)
        synced_roots.append(dst)
    return synced_roots


def main() -> None:
    start_time = time.perf_counter()
    elapsed_printed = False

    def _print_elapsed_time() -> None:
        nonlocal elapsed_printed
        if elapsed_printed:
            return
        elapsed_printed = True
        elapsed = time.perf_counter() - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed - minutes * 60
        print(
            f"Total elapsed time: {elapsed:.2f}s ({minutes}m {seconds:.2f}s)",
            flush=True,
        )

    atexit.register(_print_elapsed_time)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--parent-dir",
        type=Path,
        required=True,
        help=(
            "Directory that *contains* each dataset folder (same as --root for the upstream converter). "
            "Relative paths are resolved under LEROBOT_HOME."
        ),
    )
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        required=True,
        help="Folder names under parent-dir for each v2.1 dataset (e.g. basket_pick_0410_lerobot).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Pass through to upstream converter (default: do not push).",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Pass --force-conversion to upstream (e.g. re-run if needed).",
    )
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for merge workflow. Converted v3 datasets are first copied under this "
            "directory, and the merged dataset root is "
            "`<merge-output-dir> / <merge-repo-id>` (each path segment in merge-repo-id is a subdirectory). "
            "The parent may already exist; the leaf directory must not (unless --merge-overwrite). "
            "Relative paths are resolved under LEROBOT_HOME."
        ),
    )
    parser.add_argument(
        "--converted-output-dir",
        type=Path,
        default=Path(DEFAULT_V3_PARENT),
        help=(
            "Parent directory for converted v3 datasets. Defaults to "
            f"`{DEFAULT_V3_PARENT}` under LEROBOT_HOME. "
            "For parent-dir under `lerobot/...`, subfolders are preserved (e.g. "
            "`lerobot/desk_basket_pick` -> `lerobot_v3/desk_basket_pick`)."
        ),
    )
    parser.add_argument(
        "--merge-repo-id",
        type=str,
        default="local/merged_lerobot_v30",
        help="repo_id string for the merged dataset; also determines the output folder name under --merge-output-dir.",
    )
    parser.add_argument(
        "--merge-overwrite",
        action="store_true",
        help=(
            "Overwrite merge artifacts under --merge-output-dir: existing copied converted datasets and "
            "existing merged output leaf directory."
        ),
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only run merge (datasets must already be v3.0 under parent-dir).",
    )
    parser.add_argument(
        "--in-place-convert",
        action="store_true",
        help=(
            "Replace each v2.1 folder with v3.0 and move the old tree to *_old (legacy upstream behavior). "
            "Default is --keep-original: write v3.0 to <repo-id>_v30 and leave the v2.1 folder unchanged."
        ),
    )
    args = parser.parse_args()

    parent_dir = _resolve_data_path(args.parent_dir)
    converted_output_dir = _resolve_data_path(args.converted_output_dir)
    keep_original = not args.in_place_convert
    converted_roots: list[Path] = []
    for repo_id in args.repo_ids:
        ds_dir = parent_dir / repo_id
        if not ds_dir.is_dir():
            print(f"Missing dataset directory: {ds_dir}", file=sys.stderr)
            sys.exit(1)
        v30_dir = _v30_sibling_path(parent_dir, repo_id)
        expected_v30_dir = _expected_v30_path(parent_dir, repo_id, converted_output_dir)
        ver = _load_codebase_version(ds_dir)
        print(f"{repo_id}: codebase_version={ver}", flush=True)
        if args.skip_convert:
            if expected_v30_dir.is_dir() and _load_codebase_version(expected_v30_dir) == "v3.0":
                _postprocess_converted_dataset(expected_v30_dir)
                converted_roots.append(expected_v30_dir)
                print(f"  Merge source: {expected_v30_dir}", flush=True)
            elif v30_dir.is_dir() and _load_codebase_version(v30_dir) == "v3.0":
                converted_root = _relocate_converted_dataset(
                    v30_dir,
                    expected_v30_dir,
                    args.force_conversion,
                )
                _postprocess_converted_dataset(converted_root)
                converted_roots.append(converted_root)
                print(f"  Merge source: {converted_root}", flush=True)
            elif ver == "v3.0":
                _postprocess_converted_dataset(ds_dir)
                converted_roots.append(ds_dir)
                print(f"  Merge source (in-place v3.0): {ds_dir}", flush=True)
            else:
                print(
                    f"Refusing merge-only: expected v3.0 at {v30_dir} or in-place at {ds_dir}, "
                    f"got {ver} at source.",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif ver == "v3.0":
            print("  Skip conversion (source already v3.0).", flush=True)
            converted_root = ds_dir
            if not args.in_place_convert:
                converted_root = _relocate_converted_dataset(
                    ds_dir,
                    expected_v30_dir,
                    args.force_conversion,
                )
            _postprocess_converted_dataset(converted_root)
            converted_roots.append(converted_root)
        elif ver != "v2.1":
            print(
                f"  This script only automates v2.1 -> v3.0. For other versions, use the appropriate "
                f"LeRobot migration tools or ask upstream.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            if keep_original:
                _run_v21_to_v30_direct(
                    source_root=parent_dir,
                    target_root=expected_v30_dir,
                    repo_id=repo_id,
                    force=args.force_conversion,
                    push_to_hub=args.push_to_hub,
                )
                converted_root = expected_v30_dir
            else:
                # Keep legacy in-place behavior when explicitly requested.
                converter = _repo_root() / "src" / "lerobot" / "datasets" / "v30" / "convert_dataset_v21_to_v30.py"
                if not converter.is_file():
                    print(f"Converter not found: {converter}", file=sys.stderr)
                    sys.exit(1)
                cmd = [
                    sys.executable,
                    str(converter),
                    "--repo-id",
                    repo_id,
                    "--root",
                    str(parent_dir),
                    "--push-to-hub",
                    "true" if args.push_to_hub else "false",
                ]
                if args.force_conversion:
                    cmd.append("--force-conversion")
                print("Running:", " ".join(cmd), flush=True)
                import subprocess

                subprocess.run(cmd, check=True)
                converted_root = ds_dir
                _postprocess_converted_dataset(converted_root)
            converted_roots.append(converted_root)

    try:
        if args.merge_output_dir is not None:
            merge_parent = _resolve_data_path(args.merge_output_dir)
            merge_parent.mkdir(parents=True, exist_ok=True)
            converted_roots = _sync_merge_sources(converted_roots, merge_parent, args.merge_overwrite)
            # LeRobot merge creates a *new* dataset root; it must not exist yet. Using a subdir under
            # merge-output-dir matches lerobot_edit_dataset (root / repo_id) and avoids FileExistsError
            # when the parent folder already exists.
            out = merge_parent.joinpath(*args.merge_repo_id.split("/")).resolve()
            if out.exists():
                if args.merge_overwrite:
                    shutil.rmtree(out)
                else:
                    print(
                        f"Merged output path already exists: {out}\n"
                        "Remove it, choose a different --merge-repo-id or --merge-output-dir, "
                        "or pass --merge-overwrite.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            _merge_local(converted_roots, out, args.merge_repo_id)
    finally:
        _print_elapsed_time()


if __name__ == "__main__":
    main()
