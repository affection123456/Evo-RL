#!/usr/bin/env python3
"""Normalize EE gripper extras from [0, 1000] to [0, 1] in a local LeRobot v3 dataset.

Magazine / dual-arm EE layout used here:
  [0:14]  dual-arm xyz+quat
  [14:26] 12 gripper channels (raw 0-1000)
  [26:]   remaining extras / padding

Rewrites parquet under ``data/`` for matching feature columns. Idempotent when
values already look like [0, 1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_KEYS = (
    "observation.ee_state",
    "observation.ee_actions",
    "observation.state",
    "action",
    "ee_state",
    "ee_actions",
    "state",
    "actions",
)
GRIPPER_START = 14
GRIPPER_DIM = 12
RAW_MAX = 1000.0


def _normalize_column(values: list, *, scale: float, start: int, dim: int) -> tuple[list, bool]:
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)
    if arr.shape[-1] < start + dim:
        return values, False
    slice_ = arr[..., start : start + dim]
    # Already normalized (or empty): skip.
    if float(np.nanmax(np.abs(slice_))) <= 1.5:
        return values, False
    arr = arr.copy()
    arr[..., start : start + dim] = slice_ / scale
    return [row for row in arr], True


def normalize_dataset(
    root: Path,
    *,
    keys: tuple[str, ...],
    gripper_start: int,
    gripper_dim: int,
    scale: float,
    dry_run: bool,
) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    present = [k for k in keys if k in features]
    if not present:
        raise RuntimeError(f"None of {list(keys)} found in {info_path}")

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet under {root / 'data'}")

    print(
        f"root={root} keys={present} gripper=[{gripper_start}:{gripper_start + gripper_dim}] "
        f"/ {scale} dry_run={dry_run} files={len(data_files)}"
    )
    total_changed = 0
    for path in data_files:
        table = pq.read_table(path)
        arrays = {}
        changed_any = False
        for name in table.column_names:
            col = table.column(name)
            if name not in present:
                arrays[name] = col
                continue
            values = col.to_pylist()
            new_values, changed = _normalize_column(
                values, scale=scale, start=gripper_start, dim=gripper_dim
            )
            if changed:
                changed_any = True
                total_changed += 1
                sample = np.asarray(new_values[0])
                print(
                    f"  {path.name} {name}: scaled "
                    f"min={sample[gripper_start:gripper_start + gripper_dim].min():.4f} "
                    f"max={sample[gripper_start:gripper_start + gripper_dim].max():.4f}"
                )
                arrays[name] = pa.array(new_values)
            else:
                arrays[name] = col
        if changed_any and not dry_run:
            pq.write_table(pa.table(arrays, schema=table.schema), path)
        elif changed_any:
            print(f"  dry-run skip write {path}")

    print(f"Done. columns_scaled_across_files={total_changed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing meta/ and data/",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Feature columns to scale when present",
    )
    parser.add_argument("--gripper-start", type=int, default=GRIPPER_START)
    parser.add_argument("--gripper-dim", type=int, default=GRIPPER_DIM)
    parser.add_argument("--scale", type=float, default=RAW_MAX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    normalize_dataset(
        args.root.resolve(),
        keys=tuple(args.keys),
        gripper_start=args.gripper_start,
        gripper_dim=args.gripper_dim,
        scale=args.scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
