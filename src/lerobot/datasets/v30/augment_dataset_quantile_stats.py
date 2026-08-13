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


def process_single_episode(dataset: LeRobotDataset, episode_idx: int) -> dict:
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
        col = batch[key]
        data = np.stack([np.asarray(v) for v in col]) if isinstance(col, list) else np.asarray(col)
        keepdims = data.ndim == 1
        ep_stats[key] = get_feature_stats(
            data, axis=0, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
        )

    return ep_stats


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset) -> dict[str, dict]:
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
            episode_stats_list.append(process_single_episode(dataset, episode_idx))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(process_single_episode, dataset, episode_idx): episode_idx
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
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        push_to_hub: Push updated dataset metadata to Hugging Face Hub
    """
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

    new_stats = compute_quantile_stats_for_dataset(dataset)

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

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
