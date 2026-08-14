#!/usr/bin/env python3
"""Simulate OpenPI VLA client with training-set frames; compare pred vs GT action chunks.

Matches pi05 bridge rename (without_ref):
  observation/state -> observation.ee_state
  observation/image -> observation.images.top_head
  observation/right_wrist_image -> observation.images.hand_right

Example:
  PYTHONPATH=src:../openpi/src:../openpi/packages/openpi-client/src \\
    python scripts/sim_vla_client_dataset_eval.py \\
      --dataset-root /mnt/nas/datasets/rldata/lerobot/magazine_shelf_pick_20260807_0 \\
      --host 127.0.0.1 --port 8003 --num-samples 4 \\
      --out-dir outputs/eval/0812_pi05_magazine_shelf_sim
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy


def _as_hwc_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8) if x.max() > 1.5 else (x * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip(np.abs((a * b).sum(axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _pack_obs(sample: dict, prompt: str, *, include_left: bool = True) -> dict:
    ee = np.asarray(sample["ee_state"], dtype=np.float32).reshape(-1)
    top = _as_hwc_uint8(sample["top_head"])
    right = _as_hwc_uint8(sample["hand_right"])
    out = {
        "observation/state": ee,
        "observation/image": top,
        "observation/right_wrist_image": right,
        "prompt": prompt,
    }
    if include_left and "hand_left" in sample:
        out["observation/left_wrist_image"] = _as_hwc_uint8(sample["hand_left"])
    return out


class _WsClient:
    """Reuse one OpenPI msgpack WebSocket (metadata on connect, then infer)."""

    def __init__(self, host: str, port: int) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None, ping_interval=30, ping_timeout=600
        )
        _ = msgpack_numpy.unpackb(self._ws.recv())

    def infer(self, obs: dict) -> np.ndarray:
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(resp)
        out = msgpack_numpy.unpackb(resp)
        if not isinstance(out, dict) or "actions" not in out:
            raise RuntimeError(
                f"unexpected response keys: {list(out) if isinstance(out, dict) else type(out)}"
            )
        return np.asarray(out["actions"], dtype=np.float32)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def _align_gt_to_pred(gt: np.ndarray, pred_dim: int) -> np.ndarray:
    """Align dataset ee_actions (quat, often 34) to service pred (quat after Rot6D decode, often 28).

    Layout shared for first 14: xyz_L(3)+quat_L(4)+xyz_R(3)+quat_R(4).
    Remaining dims are gripper/hand extras; truncate/pad to pred_dim.
    """
    gt = np.asarray(gt, dtype=np.float32)
    if gt.shape[-1] == pred_dim:
        return gt
    if gt.shape[-1] > pred_dim:
        return gt[..., :pred_dim]
    pad = np.zeros((*gt.shape[:-1], pred_dim - gt.shape[-1]), dtype=np.float32)
    return np.concatenate([gt, pad], axis=-1)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    t = min(len(pred), len(gt))
    pred, gt = pred[:t], _align_gt_to_pred(gt[:t], pred.shape[-1])
    d = min(pred.shape[-1], gt.shape[-1], 26)
    pred, gt = pred[..., :d], gt[..., :d]
    left_xyz = np.linalg.norm(pred[:, 0:3] - gt[:, 0:3], axis=-1)
    right_xyz = np.linalg.norm(pred[:, 7:10] - gt[:, 7:10], axis=-1)
    left_q = _quat_angle_deg(pred[:, 3:7], gt[:, 3:7])
    right_q = _quat_angle_deg(pred[:, 10:14], gt[:, 10:14])
    grip = np.linalg.norm(pred[:, 14:26] - gt[:, 14:26], axis=-1) if d >= 26 else np.full(t, np.nan)
    return {
        "horizon": int(t),
        "left_xyz_mm_mean": float(left_xyz.mean() * 1000),
        "left_xyz_mm_p0": float(left_xyz[0] * 1000),
        "right_xyz_mm_mean": float(right_xyz.mean() * 1000),
        "right_xyz_mm_p0": float(right_xyz[0] * 1000),
        "left_q_deg_mean": float(left_q.mean()),
        "left_q_deg_p0": float(left_q[0]),
        "right_q_deg_mean": float(right_q.mean()),
        "right_q_deg_p0": float(right_q[0]),
        "grip_l2_mean": float(np.nanmean(grip)),
        "grip_l2_p0": float(grip[0]) if np.isfinite(grip[0]) else None,
        "pred_dim": int(pred.shape[-1]),
        "gt_dim": int(gt.shape[-1]),
    }


def _plot(pred: np.ndarray, gt: np.ndarray, state: np.ndarray, title: str, path: Path) -> None:
    t = min(len(pred), len(gt))
    pred, gt = pred[:t], _align_gt_to_pred(gt[:t], pred.shape[-1])
    xs = np.arange(t)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(title)

    # right xyz (client mainly uses right arm)
    for i, name in enumerate("xyz"):
        axes[0, 0].plot(xs, gt[:, 7 + i], label=f"gt_{name}", alpha=0.85)
        axes[0, 0].plot(xs, pred[:, 7 + i], "--", label=f"pred_{name}", alpha=0.85)
    axes[0, 0].axhline(state[7], color="k", ls=":", lw=0.8, alpha=0.5)
    axes[0, 0].set_title("Right EE xyz (m)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xs, np.linalg.norm(pred[:, 7:10] - gt[:, 7:10], axis=-1) * 1000, label="||pred-gt||")
    axes[0, 1].plot(xs, np.linalg.norm(pred[:, 7:10] - state[7:10], axis=-1) * 1000, label="||pred-state||")
    axes[0, 1].set_title("Right xyz error (mm)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for i, name in enumerate(["qx", "qy", "qz", "qw"]):
        axes[1, 0].plot(xs, gt[:, 10 + i], label=f"gt_{name}", alpha=0.8)
        axes[1, 0].plot(xs, pred[:, 10 + i], "--", label=f"pred_{name}", alpha=0.8)
    axes[1, 0].set_title("Right EE quat (xyzw)")
    axes[1, 0].legend(ncol=4, fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(xs, _quat_angle_deg(pred[:, 10:14], gt[:, 10:14]), label="pred vs gt")
    axes[1, 1].plot(xs, _quat_angle_deg(pred[:, 10:14], np.broadcast_to(state[10:14], (t, 4))), label="pred vs state")
    axes[1, 1].set_title("Right quat angle (deg)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # gripper: left idx14 / right idx20 (and replicated channels)
    if pred.shape[-1] >= 21 and gt.shape[-1] >= 21:
        axes[2, 0].plot(xs, gt[:, 14], label="gt_L")
        axes[2, 0].plot(xs, pred[:, 14], "--", label="pred_L")
        axes[2, 0].plot(xs, gt[:, 20], label="gt_R")
        axes[2, 0].plot(xs, pred[:, 20], "--", label="pred_R")
        axes[2, 0].set_ylim(-0.05, 1.05)
        axes[2, 0].set_title("Gripper channels [0,1] (L=14, R=20)")
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)

    # left arm xyz for completeness
    for i, name in enumerate("xyz"):
        axes[2, 1].plot(xs, gt[:, i], label=f"gt_{name}", alpha=0.8)
        axes[2, 1].plot(xs, pred[:, i], "--", label=f"pred_{name}", alpha=0.8)
    axes[2, 1].set_title("Left EE xyz (m)")
    axes[2, 1].legend(ncol=3, fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/nas/datasets/rldata/lerobot/magazine_shelf_pick_20260807_0"),
    )
    p.add_argument("--repo-id", default="magazine_shelf_pick_20260807_0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--episode", type=int, default=0, help="Base episode; samples spaced within it")
    p.add_argument("--prompt", default=None, help="Override task prompt; default from tasks.parquet")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eval/sim_vla_client"))
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.dataset_root.resolve()
    ds = LeRobotDataset(repo_id=args.repo_id, root=root, video_backend="torchcodec")
    tasks = json.loads((root / "meta" / "info.json").read_text())  # noqa: F841 — keep for debug
    # prompt from tasks table if available
    prompt = args.prompt
    if prompt is None:
        try:
            import pyarrow.parquet as pq

            task_tbl = pq.read_table(root / "meta" / "tasks.parquet")
            # LeRobot tasks.parquet: task string is usually the index column
            names = set(task_tbl.column_names)
            if "__index_level_0__" in names:
                prompt = str(task_tbl.column("__index_level_0__")[0].as_py())
            elif "task" in names:
                prompt = str(task_tbl.column("task")[0].as_py())
            else:
                # last-resort: first string-like column
                prompt = None
                for col in task_tbl.column_names:
                    val = task_tbl.column(col)[0].as_py()
                    if isinstance(val, str) and len(val) > 3:
                        prompt = val
                        break
            if not prompt:
                prompt = "Please grasp the magazine with right hand"
        except Exception:
            prompt = "Please grasp the magazine with right hand"

    # Collect candidate global indices near start of chosen episode (enough room for chunk).
    ep_meta = ds.meta.episodes[args.episode]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    usable = ep_to - ep_from - args.chunk_size
    if usable <= 0:
        raise SystemExit(f"episode {args.episode} too short: [{ep_from},{ep_to})")
    offsets = np.linspace(0, max(usable - 1, 0), num=args.num_samples, dtype=int)
    indices = [ep_from + int(o) for o in offsets]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f"dataset={root} episode={args.episode} frames=[{ep_from},{ep_to}) prompt={prompt!r}")
    print(f"service=ws://{args.host}:{args.port} samples={indices}")

    client = _WsClient(args.host, args.port)
    try:
        for k, idx in enumerate(indices):
            sample = ds[idx]
            # Build GT horizon from consecutive frames' ee_actions (absolute EE quat target).
            gt_rows = []
            for j in range(args.chunk_size):
                row = ds[idx + j]
                gt_rows.append(np.asarray(row["ee_actions"], dtype=np.float32).reshape(-1))
            gt = np.stack(gt_rows, axis=0)
            state = np.asarray(sample["ee_state"], dtype=np.float32).reshape(-1)

            sample_prompt = (
                str(sample["task"]) if ("task" in sample and sample["task"]) else prompt
            )
            obs = _pack_obs(
                {
                    "ee_state": state,
                    "top_head": sample["top_head"],
                    "hand_right": sample["hand_right"],
                    "hand_left": sample.get("hand_left"),
                },
                sample_prompt,
                include_left=True,
            )
            pred = client.infer(obs)
            gt_aligned = _align_gt_to_pred(gt, pred.shape[-1])
            m = _metrics(pred, gt_aligned)
            np.savez_compressed(
                args.out_dir / f"ep{args.episode}_sample{k}_idx{idx}.npz",
                pred=pred,
                gt=gt_aligned,
                state=state,
                prompt=np.asarray(sample_prompt),
            )
            # first-point vs state (absolute policy should be close)
            m["p0_right_xyz_vs_state_mm"] = float(np.linalg.norm(pred[0, 7:10] - state[7:10]) * 1000)
            m["p0_right_q_vs_state_deg"] = float(
                _quat_angle_deg(pred[0:1, 10:14], state[10:14][None])[0]
            )
            m["index"] = int(idx)
            m["frame_index"] = int(sample["frame_index"]) if "frame_index" in sample else None
            summary.append(m)

            plot_path = args.out_dir / f"ep{args.episode}_sample{k}_idx{idx}.png"
            _plot(
                pred,
                gt_aligned,
                state,
                title=(
                    f"ep{args.episode} idx={idx} | "
                    f"Rxyz p0={m['right_xyz_mm_p0']:.1f}mm mean={m['right_xyz_mm_mean']:.1f}mm | "
                    f"Rq p0={m['right_q_deg_p0']:.2f}° mean={m['right_q_deg_mean']:.2f}°"
                ),
                path=plot_path,
            )
            print(
                f"[{k}] idx={idx} pred={pred.shape} gt={gt_aligned.shape} "
                f"Rxyz(mm) p0={m['right_xyz_mm_p0']:.2f} mean={m['right_xyz_mm_mean']:.2f} "
                f"Rq(deg) p0={m['right_q_deg_p0']:.2f} mean={m['right_q_deg_mean']:.2f} "
                f"p0_vs_state(mm/deg)={m['p0_right_xyz_vs_state_mm']:.2f}/{m['p0_right_q_vs_state_deg']:.2f} "
                f"grip_l2 p0={m['grip_l2_p0']} plot={plot_path}"
            )
    finally:
        client.close()

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # aggregate
    keys = [
        "right_xyz_mm_p0",
        "right_xyz_mm_mean",
        "right_q_deg_p0",
        "right_q_deg_mean",
        "p0_right_xyz_vs_state_mm",
        "p0_right_q_vs_state_deg",
    ]
    print("--- aggregate ---")
    for key in keys:
        vals = [s[key] for s in summary]
        print(f"{key}: mean={np.mean(vals):.3f}  median={np.median(vals):.3f}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
