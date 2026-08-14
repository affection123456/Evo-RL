#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script augments existing LeRobot datasets with quantile statistics.

Most datasets created before the quantile feature was added do not contain
quantile statistics (q01, q10, q50, q90, q99) in their metadata. This script:

1. Loads an existing LeRobot dataset in v3.0 format
2. Checks if scalar features already contain quantile statistics
3. If missing (or with ``--overwrite``), computes quantile statistics for scalar features only
4. Updates the dataset metadata (image/video pixel stats are not written; use ImageNet at train time)

Usage:

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import concurrent.futures
import logging
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from requests import HTTPError
from tqdm import tqdm

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1)
    row1 = np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1)
    row2 = np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def _matrix_first_two_cols_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Flatten the first two rotation-matrix columns as ``[col0, col1]``."""
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _quat_pose_to_rot6d(x: np.ndarray) -> np.ndarray:
    """Convert dual-arm xyz+quat to [left pose9, right pose9, raw tail]."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] < 14:
        raise ValueError(f"Expected last dim >= 14 for dual-arm xyz+quat layout, got {x.shape}")
    left = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 3:7]))
    right = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 10:14]))
    return np.concatenate([x[..., :3], left, x[..., 7:10], right, x[..., 14:]], axis=-1)


def _pad_or_clip_last_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, dim - x.shape[-1])])
    return x


def _feature_array(batch, key: str) -> np.ndarray | None:
    if key not in batch.column_names:
        return None
    col = batch[key]
    return np.stack([np.asarray(v) for v in col]) if isinstance(col, list) else np.asarray(col)


def _quantile_list_keys() -> list[str]:
    return [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]


def _scalar_feature_keys(features: dict) -> list[str]:
    return [key for key, ft in features.items() if ft["dtype"] not in ("string", "image", "video")]


def needs_scalar_quantile_stats(features: dict, stats: dict[str, dict] | None) -> bool:
    """Return True if any non-visual feature is missing quantile statistics."""
    quantile_list_keys = _quantile_list_keys()
    if stats is None:
        return True

    for key in _scalar_feature_keys(features):
        feature_stats = stats.get(key, {})
        if not any(q_key in feature_stats for q_key in quantile_list_keys):
            return True

    return False


def _subtract_state_delta(actions_rot6d: np.ndarray, state_rot6d: np.ndarray, dims: int) -> np.ndarray:
    """actions_rot6d[..., :dims] -= state, broadcasting a trailing time/horizon dim if needed."""
    out = actions_rot6d.copy()
    if actions_rot6d.ndim == state_rot6d.ndim:
        out[..., :dims] -= state_rot6d[..., :dims]
    else:
        out[..., :dims] -= state_rot6d[..., None, :dims]
    return out


def process_single_episode(
    dataset: LeRobotDataset,
    episode_idx: int,
    *,
    pi0_dmp_rot6d_stats: bool = False,
    pi0_dmp_rot6d_delta: bool = False,
    pi05_rot6d_stats: bool = False,
    pi05_rot6d_delta: bool = False,
    rot6d_state_dim: int = 32,
    rot6d_action_dim: int = 32,
) -> dict:
    """Process a single episode and return statistics for non-visual features only.

    Image/video pixel statistics are skipped (training uses ImageNet mean/std at load time).
    """
    logging.info(f"Computing stats for episode {episode_idx}")

    dataset._ensure_hf_dataset_loaded()
    start_idx = int(dataset.meta.episodes[episode_idx]["dataset_from_index"])
    end_idx = int(dataset.meta.episodes[episode_idx]["dataset_to_index"])
    if start_idx >= end_idx:
        return {}

    scalar_keys = _scalar_feature_keys(dataset.features)
    batch = dataset.hf_dataset.select(range(start_idx, end_idx))

    ep_stats = {}
    for key in scalar_keys:
        data = _feature_array(batch, key)
        if data is None:
            continue
        keepdims = data.ndim == 1
        ep_stats[key] = get_feature_stats(
            data, axis=0, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
        )

    if pi0_dmp_rot6d_stats:
        state = _feature_array(batch, "ee_state")
        ref_state = _feature_array(batch, "ref_ee_state")
        actions = _feature_array(batch, "ee_actions")
        ref_actions = _feature_array(batch, "ref_ee_actions")

        if state is not None:
            state_rot6d = _pad_or_clip_last_dim(
                _quat_pose_to_rot6d(state),
                rot6d_state_dim,
            )
            ep_stats["observation.state"] = get_feature_stats(
                state_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )
        if ref_state is not None:
            ref_state_rot6d = _pad_or_clip_last_dim(
                _quat_pose_to_rot6d(ref_state),
                rot6d_state_dim,
            )
            ep_stats["observation.reference.state"] = get_feature_stats(
                ref_state_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )
        # Absolute Rot6D by default; optional pose-only delta (first 18 dims).
        pose_delta_dims = 18
        if actions is not None:
            actions_rot6d = _pad_or_clip_last_dim(
                _quat_pose_to_rot6d(actions),
                rot6d_action_dim,
            )
            if pi0_dmp_rot6d_delta and state is not None:
                state_rot6d = _pad_or_clip_last_dim(
                    _quat_pose_to_rot6d(state),
                    rot6d_state_dim,
                )
                dims = min(pose_delta_dims, rot6d_action_dim, rot6d_state_dim)
                actions_rot6d = actions_rot6d.copy()
                actions_rot6d[..., :dims] -= state_rot6d[:, None, :dims]
            ep_stats["action"] = get_feature_stats(
                actions_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )
        if ref_actions is not None:
            ref_actions_rot6d = _pad_or_clip_last_dim(
                _quat_pose_to_rot6d(ref_actions),
                rot6d_action_dim,
            )
            if pi0_dmp_rot6d_delta and ref_state is not None:
                ref_state_rot6d = _pad_or_clip_last_dim(
                    _quat_pose_to_rot6d(ref_state),
                    rot6d_state_dim,
                )
                dims = min(pose_delta_dims, rot6d_action_dim, rot6d_state_dim)
                ref_actions_rot6d = ref_actions_rot6d.copy()
                ref_actions_rot6d[..., :dims] -= ref_state_rot6d[:, None, :dims]
            ep_stats["observation.ref_actions"] = get_feature_stats(
                ref_actions_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )

    elif pi05_rot6d_stats:
        # pi05 datasets store EE quat under observation.ee_*; joint observation.state is unused.
        state = _feature_array(batch, "observation.ee_state")
        if state is None:
            state = _feature_array(batch, "ee_state")
        actions = _feature_array(batch, "observation.ee_actions")
        if actions is None:
            actions = _feature_array(batch, "ee_actions")

        pose_delta_dims = 18
        if state is not None:
            state_rot6d = _pad_or_clip_last_dim(_quat_pose_to_rot6d(state), rot6d_state_dim)
            ep_stats["observation.state"] = get_feature_stats(
                state_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )
        # Absolute Rot6D by default; optional pose-only delta.
        if actions is not None:
            actions_rot6d = _pad_or_clip_last_dim(_quat_pose_to_rot6d(actions), rot6d_action_dim)
            if pi05_rot6d_delta and state is not None:
                state_rot6d = _pad_or_clip_last_dim(_quat_pose_to_rot6d(state), rot6d_state_dim)
                dims = min(pose_delta_dims, rot6d_action_dim, rot6d_state_dim)
                actions_rot6d = _subtract_state_delta(actions_rot6d, state_rot6d, dims)
            ep_stats["action"] = get_feature_stats(
                actions_rot6d, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
            )

    return ep_stats


def compute_quantile_stats_for_dataset(
    dataset: LeRobotDataset,
    *,
    pi0_dmp_rot6d_stats: bool = False,
    pi0_dmp_rot6d_delta: bool = False,
    pi05_rot6d_stats: bool = False,
    pi05_rot6d_delta: bool = False,
    rot6d_state_dim: int = 32,
    rot6d_action_dim: int = 32,
) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Args:
        dataset: The LeRobot dataset to compute statistics for

    Returns:
        Dictionary containing aggregated statistics with quantiles

    Note:
        Video decoding operations are not thread-safe, so we process episodes sequentially
        when video keys are present. For datasets without videos, we use parallel processing
        with ThreadPoolExecutor for better performance.
    """
    logging.info(
        f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes "
        f"(scalar features only; skipping {len(dataset.meta.camera_keys)} camera keys)"
    )

    episode_stats_list = []
    max_workers = min(dataset.num_episodes, 16)

    if max_workers <= 1:
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
            episode_stats_list.append(
                process_single_episode(
                    dataset,
                    episode_idx,
                    pi0_dmp_rot6d_stats=pi0_dmp_rot6d_stats,
                    pi0_dmp_rot6d_delta=pi0_dmp_rot6d_delta,
                    pi05_rot6d_stats=pi05_rot6d_stats,
                    pi05_rot6d_delta=pi05_rot6d_delta,
                    rot6d_state_dim=rot6d_state_dim,
                    rot6d_action_dim=rot6d_action_dim,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(
                    process_single_episode,
                    dataset,
                    episode_idx,
                    pi0_dmp_rot6d_stats=pi0_dmp_rot6d_stats,
                    pi0_dmp_rot6d_delta=pi0_dmp_rot6d_delta,
                    pi05_rot6d_stats=pi05_rot6d_stats,
                    pi05_rot6d_delta=pi05_rot6d_delta,
                    rot6d_state_dim=rot6d_state_dim,
                    rot6d_action_dim=rot6d_action_dim,
                ): episode_idx
                for episode_idx in range(dataset.num_episodes)
            }

            episode_results = {}
            with tqdm(total=dataset.num_episodes, desc="Processing episodes") as pbar:
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_idx = future_to_episode[future]
                    episode_results[episode_idx] = future.result()
                    pbar.update(1)

        for episode_idx in range(dataset.num_episodes):
            if episode_idx in episode_results:
                episode_stats_list.append(episode_results[episode_idx])

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Aggregating statistics from {len(episode_stats_list)} episodes")
    return aggregate_stats(episode_stats_list)


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    pi0_dmp_rot6d_stats: bool = False,
    pi0_dmp_rot6d_delta: bool = False,
    pi05_rot6d_stats: bool = False,
    pi05_rot6d_delta: bool = False,
    rot6d_state_dim: int = 32,
    rot6d_action_dim: int = 32,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        push_to_hub: Push updated dataset metadata to Hugging Face Hub
        pi0_dmp_rot6d_stats: Write DMP absolute Rot6D stats from ee_* / ref_* keys
        pi0_dmp_rot6d_delta: If True with pi0_dmp_rot6d_stats, use pose-only delta for actions
        pi05_rot6d_stats: Write pi05 absolute Rot6D stats from observation.ee_* keys
        pi05_rot6d_delta: If True with pi05_rot6d_stats, use pose-only delta for action stats
    """
    if pi0_dmp_rot6d_stats and pi05_rot6d_stats:
        raise ValueError("Use only one of pi0_dmp_rot6d_stats / pi05_rot6d_stats")
    root_path = Path(root).expanduser().resolve() if root is not None else None
    # Local datasets under ``--root`` are opened by absolute path; using a Hub-style ``org/name`` as
    # ``repo_id`` still triggers ``get_safe_version`` → Hub on cache miss (breaks offline). Use the
    # on-disk folder name as ``repo_id`` when ``root`` is set.
    load_repo_id = root_path.name if root_path is not None else repo_id
    logging.info(f"Loading dataset: {repo_id}" + (f" (local root={root_path}, load_repo_id={load_repo_id})" if root_path else ""))
    dataset = LeRobotDataset(
        repo_id=load_repo_id,
        root=str(root_path) if root_path is not None else None,
        download_videos=False if root_path is not None else True,
    )

    if not overwrite and not needs_scalar_quantile_stats(dataset.features, dataset.meta.stats):
        logging.info("Scalar features already contain quantile statistics. No action needed.")
        return

    logging.info("Computing quantile statistics for scalar features...")

    new_stats = compute_quantile_stats_for_dataset(
        dataset,
        pi0_dmp_rot6d_stats=pi0_dmp_rot6d_stats,
        pi0_dmp_rot6d_delta=pi0_dmp_rot6d_delta,
        pi05_rot6d_stats=pi05_rot6d_stats,
        pi05_rot6d_delta=pi05_rot6d_delta,
        rot6d_state_dim=rot6d_state_dim,
        rot6d_action_dim=rot6d_action_dim,
    )

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    if not push_to_hub:
        logging.info("Local mode enabled: skip push_to_hub.")
        return

    dataset.push_to_hub()

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        logging.info(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        pass
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=None, repo_type="dataset")


def main():
    """Main function to run the augmentation script."""
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with quantile statistics")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht')",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Local root directory for the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile statistics if they already exist",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push updated dataset metadata to Hugging Face Hub (default: local-only).",
    )
    parser.add_argument(
        "--pi0-dmp-rot6d-stats",
        action="store_true",
        help="Also write PI0-DMP absolute Rot6D stats for state/ref_state and action/ref_actions.",
    )
    parser.add_argument(
        "--pi0-dmp-rot6d-delta",
        action="store_true",
        help="With --pi0-dmp-rot6d-stats, write pose-only delta action/ref_actions stats instead of absolute.",
    )
    parser.add_argument(
        "--pi05-rot6d-stats",
        action="store_true",
        help=(
            "Also write pi05 absolute Rot6D stats for observation.state/action from "
            "observation.ee_state / observation.ee_actions (does not affect pi0_dmp)."
        ),
    )
    parser.add_argument(
        "--pi05-rot6d-delta",
        action="store_true",
        help="With --pi05-rot6d-stats, write pose-only delta action stats instead of absolute.",
    )
    parser.add_argument("--rot6d-state-dim", type=int, default=32)
    parser.add_argument("--rot6d-action-dim", type=int, default=32)

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        pi0_dmp_rot6d_stats=args.pi0_dmp_rot6d_stats,
        pi0_dmp_rot6d_delta=args.pi0_dmp_rot6d_delta,
        pi05_rot6d_stats=args.pi05_rot6d_stats,
        pi05_rot6d_delta=args.pi05_rot6d_delta,
        rot6d_state_dim=args.rot6d_state_dim,
        rot6d_action_dim=args.rot6d_action_dim,
    )


if __name__ == "__main__":
    main()
