#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""WebSocket **test client** for ``lerobot-policy-infer``.

Loads one frame from a local LeRobot v3 dataset (same tensors as training ``__getitem__``), builds the
``observation`` payload expected by the infer server, and optionally appends the default **positive ACP**
tag to task prompts (does not read ACP labels from dataset fields).

Example::

    lerobot-policy-infer-client \
      --ws-uri=ws://127.0.0.1:8004 \
      --policy.path=/mnt/nas/wanghao/openpi_05/Evo-RL/outputs/train/0512_1/checkpoints/last \
      --dataset.root=/mnt/nas/wanghao/data/lerobot_v3/desk_basket_pick \
      --dataset.repo-id=basket_pick_0428_s_38 \
      --frame-index=0

``--dataset.root`` may be either the **parent** of the dataset folder (as in many training scripts) or the
**full** path to the dataset (the directory that contains ``meta/``). If the parent is passed, this client
resolves ``root / repo_id`` automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import parser as cfg_parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.scripts.lerobot_policy_infer import _resolve_pretrained_model_path
from lerobot.utils.import_utils import register_third_party_plugins


def _resolve_dataset_local_root(root: Path, repo_id: str) -> Path:
    """Match how people pass ``--dataset.root`` in training scripts vs ``LeRobotDataset`` API.

    ``LeRobotDataset(..., root=R)`` uses ``R`` as the **dataset root** when ``R`` is set; it does **not**
    append ``repo_id`` (unlike the default ``HF_LEROBOT_HOME / repo_id`` when ``root`` is None). So if you
    pass the parent directory plus ``--dataset.repo-id=folder``, we resolve ``root/repo_id`` when
    ``root/meta/info.json`` is missing.
    """
    root = root.expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    candidates = [root / repo_id, root.joinpath(*repo_id.split("/"))]
    for c in candidates:
        if (c / "meta" / "info.json").is_file():
            logging.info("Resolved dataset root to %s (LeRobotDataset expects full dataset dir as root)", c)
            return c
    raise FileNotFoundError(
        f"No meta/info.json under {root} or {root / repo_id}. "
        "Pass --dataset.root as the directory that **contains** meta/ (often .../<org>/<dataset_name>), "
        "or as the parent of that folder together with --dataset.repo-id."
    )


def _tensor_to_hwc_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    t = img.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(t.shape)}")
    if t.shape[0] in (1, 3) and t.dtype.is_floating_point:
        return (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    if t.shape[-1] in (1, 3, 4) and t.dtype == torch.uint8:
        arr = t.numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr)
    raise ValueError(f"Unsupported image tensor: dtype={t.dtype} shape={tuple(t.shape)}")


def _jpeg_b64_hwc(arr: np.ndarray, *, quality: int) -> str:
    rgb = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr, 3, axis=-1)
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize_hwc_to_policy(arr: np.ndarray, ft: Any) -> np.ndarray:
    """Resize HWC uint8 to policy visual feature (CHW shape: C,H,W)."""
    shape = getattr(ft, "shape", None)
    if not shape or len(shape) != 3:
        return arr
    target_h, target_w = int(shape[1]), int(shape[2])
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    pil = Image.fromarray(arr[..., :3], mode="RGB")
    return np.asarray(pil.resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)


def _acp_conditioned_task(
    *,
    base_task: str,
    enable: bool,
) -> str:
    if not enable:
        return base_task
    # In inference client, ACP prompt does not depend on dataset labels.
    # Default to positive ACP tag.
    return build_acp_tagged_task(base_task, is_positive=True)


def _build_observation_json(
    item: dict[str, Any],
    policy_cfg: PreTrainedConfig,
    rename_map: dict[str, str],
    *,
    jpeg_quality: int,
) -> dict[str, Any]:
    inv = {new: old for old, new in rename_map.items()}
    obs: dict[str, Any] = {}
    for policy_key, ft in (policy_cfg.input_features or {}).items():
        if ft.type not in (FeatureType.STATE, FeatureType.VISUAL, FeatureType.LANGUAGE):
            continue
        client_key = inv.get(policy_key, policy_key)
        if client_key not in item:
            raise KeyError(
                f"Dataset frame missing {client_key!r} for policy feature {policy_key!r}. "
                "Check dataset vs checkpoint input_features / --rename-map-json."
            )
        val = item[client_key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {client_key}, got {type(val)}")
        if ft.type == FeatureType.VISUAL or "image" in policy_key or "image" in client_key:
            hwc = _tensor_to_hwc_uint8_rgb(val)
            hwc = _resize_hwc_to_policy(hwc, ft)
            obs[client_key] = _jpeg_b64_hwc(hwc, quality=jpeg_quality)
        elif ft.type == FeatureType.LANGUAGE:
            flat = val.detach().cpu().reshape(-1)
            if flat.dtype in (torch.int32, torch.int64, torch.bool):
                obs[client_key] = [int(x) for x in flat.tolist()]
            else:
                obs[client_key] = [float(x) for x in flat.tolist()]
        else:
            obs[client_key] = [float(x) for x in val.detach().cpu().reshape(-1).tolist()]
    return obs


def _load_policy_config(policy_path: str) -> PreTrainedConfig:
    resolved = _resolve_pretrained_model_path(policy_path)
    cli_overrides = cfg_parser.get_cli_overrides("policy")
    cfg = PreTrainedConfig.from_pretrained(resolved, cli_overrides=cli_overrides)
    cfg.pretrained_path = resolved
    return cfg


async def _run_client(
    *,
    ws_uri: str,
    policy_path: str,
    dataset_root: Path | None,
    dataset_repo_id: str | None,
    frame_index: int,
    num_frames: int,
    rename_map: dict[str, str],
    acp: bool,
    spec_only: bool,
    jpeg_quality: int,
    max_size_mb: int,
    connect_retry_seconds: float,
    connect_max_attempts: int,
) -> None:
    websockets = __import__("websockets")
    max_size = max_size_mb * 1024 * 1024
    policy_cfg = _load_policy_config(policy_path)

    async def _connect_with_retry():
        attempt = 0
        while True:
            attempt += 1
            try:
                return await websockets.connect(ws_uri, max_size=max_size)
            except OSError as exc:
                if connect_max_attempts > 0 and attempt >= connect_max_attempts:
                    raise RuntimeError(
                        f"Failed to connect to {ws_uri} after {attempt} attempts."
                    ) from exc
                logging.warning(
                    "Connect to %s failed (attempt %d): %s; retrying in %.1fs",
                    ws_uri,
                    attempt,
                    exc,
                    connect_retry_seconds,
                )
                await asyncio.sleep(connect_retry_seconds)

    if spec_only:
        async with await _connect_with_retry() as ws:
            await ws.send(json.dumps({"type": "spec"}))
            logging.info("spec response: %s", await ws.recv())
        return

    if dataset_root is None or not dataset_repo_id:
        raise ValueError("--dataset.root and --dataset.repo-id are required unless --spec-only.")

    dataset_path = _resolve_dataset_local_root(dataset_root, dataset_repo_id)
    ds = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_path, download_videos=True)
    if num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    end = frame_index + num_frames
    if frame_index < 0 or end > len(ds):
        raise IndexError(f"frame range [{frame_index}, {end}) out of range for len={len(ds)}")

    async with await _connect_with_retry() as ws:
        for idx in range(frame_index, end):
            item = ds[idx]
            base_task = item["task"] if isinstance(item["task"], str) else str(item["task"])
            task = _acp_conditioned_task(
                base_task=base_task,
                enable=acp,
            )
            print(f"frame {idx} task: {task}", flush=True)
            observation = _build_observation_json(
                item, policy_cfg, rename_map, jpeg_quality=jpeg_quality
            )
            payload = json.dumps({"task": task, "observation": observation})
            logging.info(
                "frame %d: sending infer request (%d bytes JSON, JPEG q=%d)",
                idx,
                len(payload),
                jpeg_quality,
            )
            await ws.send(payload)
            raw = await ws.recv()
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                logging.info("frame %d infer response (non-JSON): %s", idx, raw)
                continue

            if response.get("ok", False):
                action = response.get("action")
                logging.info("frame %d action: %s", idx, action)
                print(f"frame {idx} action: {action}", flush=True)
            else:
                logging.warning("frame %d infer error: %s", idx, response.get("error", response))
                print(f"frame {idx} infer error: {response.get('error', response)}", file=sys.stderr, flush=True)


def main() -> None:
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ws-uri", type=str, default="ws://127.0.0.1:8765")
    ap.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    ap.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None)
    ap.add_argument("--dataset.repo-id", dest="dataset_repo_id", type=str, default=None)
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Send this many consecutive frames: infer -> wait response -> next frame (default 1).",
    )
    ap.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for image fields (smaller than PNG).")
    ap.add_argument("--max-size-mb", type=int, default=32, help="WebSocket max message size (MiB), client and server should match.")
    ap.add_argument(
        "--connect-retry-seconds",
        type=float,
        default=2.0,
        help="Retry interval (seconds) when websocket server is not ready.",
    )
    ap.add_argument(
        "--connect-max-attempts",
        type=int,
        default=0,
        help="Max websocket connect attempts (0 means retry forever).",
    )
    ap.add_argument("--rename-map-json", type=str, default="")
    ap.add_argument("--acp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--spec-only", action="store_true")
    args = ap.parse_args()

    if not args.spec_only and (args.dataset_root is None or args.dataset_repo_id is None):
        ap.error("--dataset.root and --dataset.repo-id are required unless --spec-only")

    rename_map: dict[str, str] = {}
    if args.rename_map_json.strip():
        rename_map = json.loads(args.rename_map_json)

    asyncio.run(
        _run_client(
            ws_uri=args.ws_uri,
            policy_path=args.policy_path,
            dataset_root=args.dataset_root,
            dataset_repo_id=args.dataset_repo_id,
            frame_index=args.frame_index,
            num_frames=args.num_frames,
            rename_map=rename_map,
            acp=bool(args.acp),
            spec_only=args.spec_only,
            jpeg_quality=args.jpeg_quality,
            max_size_mb=args.max_size_mb,
            connect_retry_seconds=args.connect_retry_seconds,
            connect_max_attempts=args.connect_max_attempts,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
