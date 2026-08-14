#!/usr/bin/env python3
"""
Render top_head (or other camera) episode videos with value curve overlay.

This wraps ``lerobot.scripts.value_infer_viz._export_overlay_videos`` and is the
same visualization used for Evo-RL 0515_1 OOD evaluation: original camera video
with a dynamically updating value curve drawn on top.

Prerequisites:
  - Value inference must already have written predictions into dataset parquet,
    e.g. ``complementary_info.value_0515_1_ood``.

Example:
  python -m lerobot.datasets.render_value_overlay_on_video \\
      --dataset-root /mnt/nas/wanghao/data/lerobot_v3/desk_basket_place/basket_place_a02_04_s_116_f_24 \\
      --repo-id desk_basket_place/basket_place_a02_04_s_116_f_24 \\
      --episodes 0,116 \\
      --value-field complementary_info.value_0515_1_ood \\
      --video-key observation.images.top_head \\
      --output-dir outputs/value_infer/0515_1_ood_a02_04_s10_f10_pyav/value/viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lerobot.datasets.lerobot_dataset as ld
import lerobot.datasets.video_utils as vu
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.value_infer_viz import _export_overlay_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera video with value curve overlay (Evo-RL value_infer_viz style)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Absolute path to LeRobot dataset root.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo id, e.g. desk_basket_place/basket_place_a02_04_s_116_f_24.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episodes to render: 'all', comma list, or ranges like '0-9,116-125'.",
    )
    parser.add_argument(
        "--value-field",
        type=str,
        default="complementary_info.value_0515_1_ood",
        help="Parquet column with predicted values.",
    )
    parser.add_argument(
        "--advantage-field",
        type=str,
        default="complementary_info.advantage_0515_1_ood",
        help="Parquet column with advantage values (optional overlay metadata).",
    )
    parser.add_argument(
        "--indicator-field",
        type=str,
        default="complementary_info.acp_indicator_0515_1_ood",
        help="Parquet column with ACP indicator values.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default="observation.images.top_head",
        help="Camera feature key to render.",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        default=None,
        help="Comma-separated camera keys for multiview mode. Overrides --video-key.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write overlay mp4 files.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Video decoder backend for LeRobotDataset (default: pyav).",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Output video codec passed to value_infer_viz.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Savitzky-Golay smoothing window for value curve (1 = disabled).",
    )
    parser.add_argument(
        "--frame-storage-mode",
        type=str,
        default="memory",
        choices=["memory", "disk"],
        help="Frame buffering mode inside value_infer_viz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--download-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LeRobotDataset should download missing videos.",
    )
    return parser.parse_args()


def _parse_episode_list(episodes_arg: str) -> list[int] | None:
    value = episodes_arg.strip().lower()
    if value == "all":
        return None

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))
    return sorted(parsed)


def main() -> None:
    args = parse_args()

    # Force pyav unless user overrides; avoids torchcodec/FFmpeg dependency issues.
    vu.get_safe_default_codec = lambda: args.video_backend
    ld.get_safe_default_codec = lambda: args.video_backend

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_list = _parse_episode_list(args.episodes)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=str(dataset_root),
        episodes=episode_list,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )

    written = _export_overlay_videos(
        dataset=dataset,
        value_field=args.value_field,
        advantage_field=args.advantage_field,
        indicator_field=args.indicator_field,
        viz_episodes=args.episodes,
        video_key=args.video_key,
        video_keys=args.video_keys,
        output_dir=output_dir,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
        frame_storage_mode=args.frame_storage_mode,
        smooth_window=args.smooth_window,
    )

    print(f"Rendered {len(written)} video(s) to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
