#!/usr/bin/env python3
"""
Export episode video with three camera views in parallel (left | head | right),
and overlay a dynamically updating value line chart directly on top of the video frame.

Supports:

- **Image-in-parquet**: columns like ``observation.images.hand_left`` with decodable image cells.
- **Video-on-disk (LeRobot v3.0)**: no image columns in ``data/*.parquet``; cameras are ``dtype: video``
  in ``meta/info.json``. Frames are decoded via ``LeRobotDataset`` (same timestamps as training).

Usage:
python scripts/plot_value_on_video.py --dataset /path/to/dataset \\
      --ep 0 --tag recap --smooth-window 9 --chart-alpha 0.22
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3-view episode video with dynamic value curve.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset root path.")
    parser.add_argument("--root", type=str, default="/mnt/nas/wanghao/data/lerobot_v3", help="Dataset parent dir.")
    parser.add_argument("--repo-id", type=str, default=None, help="Repo id under --root.")
    parser.add_argument("--ep", type=int, default=0, help="Episode index.")
    parser.add_argument("--tag", type=str, default="recap", help="Value tag suffix.")
    parser.add_argument("--fps", type=int, default=15, help="Output video fps.")
    parser.add_argument("--cam-height", type=int, default=360, help="Height of each camera panel.")
    parser.add_argument("--chart-alpha", type=float, default=0.22, help="Overlay opacity in [0, 1].")
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving average window for value curve.")
    parser.add_argument("--save-dir", type=str, default="outputs/plots/value_videos", help="Output directory.")
    parser.add_argument("--left-col", type=str, default=None, help="Explicit left camera: parquet column (image mode) or video feature key from info.json (video mode).")
    parser.add_argument("--head-col", type=str, default=None, help="Explicit head camera column / video feature key.")
    parser.add_argument("--right-col", type=str, default=None, help="Explicit right camera column / video feature key.")
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Decoder for video datasets (passed to LeRobotDataset). Default pyav for broad PyTorch compatibility.",
    )
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset:
        root = Path(args.dataset).expanduser().resolve()
    else:
        if not args.repo_id:
            raise ValueError("Provide either --dataset or (--root and --repo-id).")
        root = (Path(args.root).expanduser() / args.repo_id).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def decode_image(cell: Any) -> np.ndarray:
    """
    Decode parquet image cell to BGR ndarray.
    Supports:
      - dict with {'bytes': ...} (common in v3 parquet)
      - raw bytes / bytearray
      - ndarray in HWC or CHW
      - file path string
    """
    if isinstance(cell, dict):
        if "bytes" in cell:
            data = cell["bytes"]
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes.")
            return img
        if "path" in cell:
            img = cv2.imread(str(cell["path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image path: {cell['path']}")
            return img

    if isinstance(cell, (bytes, bytearray)):
        arr = np.frombuffer(cell, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode raw image bytes.")
        return img

    if isinstance(cell, str):
        img = cv2.imread(cell, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from path: {cell}")
        return img

    if isinstance(cell, np.ndarray):
        img = cell
        if img.ndim != 3:
            raise ValueError(f"Unsupported ndarray image ndim={img.ndim}.")
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        raise ValueError("Unsupported ndarray channel layout.")

    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def _image_parquet_columns(columns: list[str]) -> list[str]:
    """Columns that typically hold per-frame image payloads in image-in-parquet datasets."""
    return sorted(c for c in columns if c.startswith("observation.images."))


def load_video_feature_keys(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feats = info.get("features") or {}
    return sorted(k for k, spec in feats.items() if isinstance(spec, dict) and spec.get("dtype") == "video")


def pick_camera_feature_key(
    video_keys: list[str],
    preferred: str | None,
    aliases: list[str],
    view_name: str,
    *,
    name_hints: list[str],
) -> str:
    """Resolve a camera *feature* name (same strings as parquet columns when images are embedded)."""
    if preferred:
        if preferred not in video_keys:
            raise KeyError(f"Explicit video feature {preferred!r} for {view_name} not found in info.json.")
        return preferred
    for c in aliases:
        if c in video_keys:
            return c
    for hint in name_hints:
        for c in video_keys:
            if hint in c:
                return c
    raise KeyError(
        f"Cannot resolve video camera for {view_name}. Tried aliases {aliases!r} and name_hints {name_hints!r}. "
        f"Available video features: {video_keys}"
    )


def pick_camera_column(
    columns: list[str],
    preferred: str | None,
    aliases: list[str],
    view_name: str,
    *,
    name_hints: list[str],
) -> str:
    if preferred:
        if preferred not in columns:
            raise KeyError(f"Explicit column {preferred!r} for {view_name} not found.")
        return preferred
    for c in aliases:
        if c in columns:
            return c
    candidates = _image_parquet_columns(columns)
    for hint in name_hints:
        for c in candidates:
            if hint in c:
                return c
    avail = ", ".join(candidates) if candidates else "(none — dataset may be video-on-disk only; parquet has no image columns)"
    raise KeyError(
        f"Cannot resolve camera column for {view_name}. Tried aliases {aliases!r} and "
        f"name_hints {name_hints!r}. Available observation.images.* columns: {avail}"
    )


def chw_float_rgb_to_bgr(img: Any) -> np.ndarray:
    """Convert LeRobot video/image tensor (C,H,W float in ~[0,1], RGB) to uint8 BGR for OpenCV."""
    import torch

    if isinstance(img, torch.Tensor):
        t = img.detach().float().cpu()
        if t.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")
        if float(t.max()) <= 1.0 + 1e-3:
            t = (t * 255.0).clamp(0, 255)
        arr = t.byte().numpy()
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim != 3:
            raise ValueError(f"Expected CHW ndarray, got shape {arr.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels after CHW->HWC, got shape {arr.shape}")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < 3:
        return values.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1)
    if w < 3:
        return values.copy()
    kernel = np.ones(w, dtype=np.float32) / float(w)
    pad = w // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def overlay_value_chart(
    base_frame: np.ndarray,
    values: np.ndarray,
    current_idx: int,
    frame_index: int,
    ep: int,
    tag: str,
    alpha: float,
) -> np.ndarray:
    h, w = base_frame.shape[:2]
    canvas = base_frame.copy()
    _ = alpha

    left_pad = max(12, w // 96)
    right_pad = left_pad
    top_pad = max(8, h // 72)
    bottom_pad = max(24, h // 36)
    x0, y0 = left_pad, top_pad
    x1, y1 = w - right_pad, h - bottom_pad
    if y1 <= y0 + 8:
        return canvas

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        vmax = vmin + 1.0
    pad = (vmax - vmin) * 0.05
    y_min, y_max = vmin - pad, vmax + pad

    n = len(values)
    cx = int(round(x0 + (x1 - x0) * (min(current_idx, n - 1) / max(1, n - 1))))
    cv2.line(canvas, (cx, y0), (cx, y1), (235, 235, 235), 1)

    pts: list[tuple[int, int]] = []
    for i in range(current_idx + 1):
        x = int(x0 + i / max(n - 1, 1) * (x1 - x0))
        y_norm = np.clip((float(values[i]) - y_min) / max(1e-6, (y_max - y_min)), 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * (y1 - y0)))
        pts.append((x, y))
    if len(pts) >= 2:
        curve_width = max(2, w // 600)
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (100, 200, 255), curve_width)

    if pts:
        px, py = pts[-1]
        radius = max(4, w // 220)
        cv2.circle(canvas, (px, py), radius + 2, (120, 220, 255), 1)
        cv2.circle(canvas, (px, py), radius, (110, 240, 255), -1)

        value_text = f"{values[current_idx]:.4f}"
        tx = min(px + 10, w - 150)
        ty = max(y0 + 18, py - 12)
        cv2.putText(canvas, value_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(canvas, value_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 250, 255), 1, cv2.LINE_AA)

    title = f"Episode {ep} | Value ({tag}) | frame_index={frame_index}"
    cv2.putText(canvas, title, (x0 + 2, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (x0 + 2, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)

    frame_text = f"frame {current_idx}/{n - 1}"
    (tw, _th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    fx, fy = w - tw - 10, h - 8
    cv2.putText(canvas, frame_text, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, frame_text, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def add_panel_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _collect_episode_scalar_frames(
    parquet_files: list[Path], ep: int, columns: list[str]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    missing: set[str] = set()
    for p in parquet_files:
        try:
            part = pd.read_parquet(p, columns=columns)
        except Exception:
            fb = pd.read_parquet(p)
            for c in columns:
                if c not in fb.columns:
                    missing.add(c)
            continue
        part = part[part["episode_index"] == ep]
        if not part.empty:
            parts.append(part)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not parts:
        raise ValueError(f"Episode {ep} not found in dataset parquet files.")
    return pd.concat(parts, ignore_index=True).sort_values("frame_index").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args)
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root / 'data'}")

    value_col = f"complementary_info.value_{args.tag}"
    left_aliases = ["observation.images.left_camera", "observation.images.hand_left"]
    head_aliases = ["observation.images.head", "observation.images.top_head"]
    right_aliases = ["observation.images.right_camera", "observation.images.hand_right"]
    left_hints = ["hand_left", "left_camera", "camera_left", "left_wrist", "wrist_left", "eye_in_hand"]
    head_hints = ["top_head", "head", "overhead", "agentview", "front", "global"]
    right_hints = ["hand_right", "right_camera", "camera_right", "right_wrist", "wrist_right"]

    probe_df = pd.read_parquet(parquet_files[0])
    cols = list(probe_df.columns)
    if value_col not in cols:
        raise KeyError(f"Missing value column {value_col!r}. Available columns: {cols}")

    image_cols = _image_parquet_columns(cols)
    video_keys = load_video_feature_keys(dataset_root)
    if not image_cols and not video_keys:
        raise ValueError(
            "Cannot render cameras: data parquet has no observation.images.* columns and "
            "meta/info.json has no dtype=video features. Check dataset path and info.json."
        )
    use_video = len(image_cols) == 0 and len(video_keys) > 0

    if use_video:
        left_key = pick_camera_feature_key(
            video_keys, args.left_col, left_aliases, "left_camera", name_hints=left_hints
        )
        head_key = pick_camera_feature_key(video_keys, args.head_col, head_aliases, "head", name_hints=head_hints)
        right_key = pick_camera_feature_key(
            video_keys, args.right_col, right_aliases, "right_camera", name_hints=right_hints
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(
            repo_id=dataset_root.name,
            root=str(dataset_root),
            download_videos=False,
            video_backend=args.video_backend,
        )
        ep = int(args.ep)
        if ep < 0 or ep >= ds.meta.total_episodes:
            raise ValueError(f"Episode {ep} out of range [0, {ds.meta.total_episodes - 1}]")

        ep_meta = ds.meta.episodes[ep]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        n_frames = to_idx - from_idx

        ep_df = _collect_episode_scalar_frames(parquet_files, ep, ["episode_index", "frame_index", value_col])
        if len(ep_df) != n_frames:
            raise ValueError(
                f"Episode {ep} length mismatch: parquet rows={len(ep_df)} vs "
                f"dataset index span={n_frames} (dataset_from_index..dataset_to_index)."
            )

        values = ep_df[value_col].astype(np.float32).to_numpy()
        values = _smooth_1d(values, args.smooth_window)
        frame_indices = ep_df["frame_index"].to_numpy()

        sample_item = ds[from_idx]
        first_left = chw_float_rgb_to_bgr(sample_item[left_key])
        first_head = chw_float_rgb_to_bgr(sample_item[head_key])
        first_right = chw_float_rgb_to_bgr(sample_item[right_key])

        cam_h = args.cam_height
        left_w = int(first_left.shape[1] / first_left.shape[0] * cam_h)
        head_w = int(first_head.shape[1] / first_head.shape[0] * cam_h)
        right_w = int(first_right.shape[1] / first_right.shape[0] * cam_h)
        canvas_w = left_w + head_w + right_w
        canvas_h = cam_h

        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"value_video_ep{args.ep}_{args.tag}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (canvas_w, canvas_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

        print(f"Dataset: {dataset_root} (video mode, backend={args.video_backend})")
        print(f"Episode: {args.ep}, frames: {n_frames}")
        print(f"Cameras: left={left_key}, head={head_key}, right={right_key}")
        print(f"Output: {out_path}")

        for i, idx in enumerate(range(from_idx, to_idx)):
            item = ds[idx]
            left_img = cv2.resize(
                chw_float_rgb_to_bgr(item[left_key]), (left_w, cam_h), interpolation=cv2.INTER_AREA
            )
            head_img = cv2.resize(
                chw_float_rgb_to_bgr(item[head_key]), (head_w, cam_h), interpolation=cv2.INTER_AREA
            )
            right_img = cv2.resize(
                chw_float_rgb_to_bgr(item[right_key]), (right_w, cam_h), interpolation=cv2.INTER_AREA
            )

            left_img = add_panel_title(left_img, "left_camera")
            head_img = add_panel_title(head_img, "head")
            right_img = add_panel_title(right_img, "right_camera")
            camera_row = cv2.hconcat([left_img, head_img, right_img])

            composed = overlay_value_chart(
                base_frame=camera_row,
                values=values,
                current_idx=i,
                frame_index=int(frame_indices[i]),
                ep=args.ep,
                tag=args.tag,
                alpha=float(np.clip(args.chart_alpha, 0.0, 1.0)),
            )
            writer.write(composed)

            if (i + 1) % 100 == 0 or i == n_frames - 1:
                print(f"Rendered {i + 1}/{n_frames} frames")

        writer.release()
        print("Done.")
        print(f"Saved video: {out_path}")
        return

    # --- Image-in-parquet path ---
    left_col = pick_camera_column(cols, args.left_col, left_aliases, "left_camera", name_hints=left_hints)
    head_col = pick_camera_column(cols, args.head_col, head_aliases, "head", name_hints=head_hints)
    right_col = pick_camera_column(cols, args.right_col, right_aliases, "right_camera", name_hints=right_hints)

    required_cols = ["episode_index", "frame_index", value_col, left_col, head_col, right_col]
    parts: list[pd.DataFrame] = []
    missing_cols: set[str] = set()
    for p in parquet_files:
        try:
            part = pd.read_parquet(p, columns=required_cols)
        except Exception:
            fallback = pd.read_parquet(p)
            for c in required_cols:
                if c not in fallback.columns:
                    missing_cols.add(c)
            continue
        part = part[part["episode_index"] == args.ep]
        if not part.empty:
            parts.append(part)

    if missing_cols:
        raise KeyError(f"Missing required columns: {sorted(missing_cols)}")
    if not parts:
        raise ValueError(f"Episode {args.ep} not found in dataset: {dataset_root}")

    ep_df = pd.concat(parts, ignore_index=True).sort_values("frame_index").reset_index(drop=True)
    values = ep_df[value_col].astype(np.float32).to_numpy()
    values = _smooth_1d(values, args.smooth_window)
    frame_indices = ep_df["frame_index"].to_numpy()

    first_left = decode_image(ep_df[left_col].iloc[0])
    first_head = decode_image(ep_df[head_col].iloc[0])
    first_right = decode_image(ep_df[right_col].iloc[0])

    cam_h = args.cam_height
    left_w = int(first_left.shape[1] / first_left.shape[0] * cam_h)
    head_w = int(first_head.shape[1] / first_head.shape[0] * cam_h)
    right_w = int(first_right.shape[1] / first_right.shape[0] * cam_h)
    canvas_w = left_w + head_w + right_w
    canvas_h = cam_h

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"value_video_ep{args.ep}_{args.tag}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    print(f"Dataset: {dataset_root} (image-in-parquet mode)")
    print(f"Episode: {args.ep}, frames: {len(ep_df)}")
    print(f"Cameras: left={left_col}, head={head_col}, right={right_col}")
    print(f"Output: {out_path}")

    for i, row in ep_df.iterrows():
        left_img = cv2.resize(decode_image(row[left_col]), (left_w, cam_h), interpolation=cv2.INTER_AREA)
        head_img = cv2.resize(decode_image(row[head_col]), (head_w, cam_h), interpolation=cv2.INTER_AREA)
        right_img = cv2.resize(decode_image(row[right_col]), (right_w, cam_h), interpolation=cv2.INTER_AREA)

        left_img = add_panel_title(left_img, "left_camera")
        head_img = add_panel_title(head_img, "head")
        right_img = add_panel_title(right_img, "right_camera")
        camera_row = cv2.hconcat([left_img, head_img, right_img])

        composed = overlay_value_chart(
            base_frame=camera_row,
            values=values,
            current_idx=i,
            frame_index=int(frame_indices[i]),
            ep=args.ep,
            tag=args.tag,
            alpha=float(np.clip(args.chart_alpha, 0.0, 1.0)),
        )
        writer.write(composed)

        if (i + 1) % 100 == 0 or i == len(ep_df) - 1:
            print(f"Rendered {i + 1}/{len(ep_df)} frames")

    writer.release()
    print("Done.")
    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    main()
