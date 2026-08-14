"""Shared EE state/action layout and model-facing dimension selection.

Raw A2D EE layout (state and action use the same leading fields)::

    0:3    left xyz
    3:7    left quaternion (x, y, z, w)
    7:10   right xyz
    10:14  right quaternion (x, y, z, w)
    14:20  left gripper / dexterous-hand channels (6)
    20:26  right gripper / dexterous-hand channels (6)
    State (44D):
      26:28 head joints, 28:32 waist joints, 32:44 end wrench
    Action (34D):
      26:28 head command, 28:32 waist command, 32:34 velocity extras

The model-facing contract selects ``left``, ``right`` or ``both`` arms and the
first N gripper channels for every selected arm. Per selected arm:

* ``use_rot6d=False``: xyz(3) + quaternion(4) + gripper(N) = 7 + N
* ``use_rot6d=True``:  xyz(3) + Rot6D(6) + gripper(N) = 9 + N

The default is right-only with one gripper channel: 8D quaternion or 10D
Rot6D. Model heads may remain padded (normally 32D), but loss must only use
``physical_dim``.

For ``both``, model-facing channels are ordered as all selected poses first,
then selected grippers: ``left_pose, right_pose, left_gripper[:N],
right_gripper[:N]``. This keeps pose-only delta channels contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

EEArmMode = Literal["left", "right", "both"]

EE_LEFT_XYZ = slice(0, 3)
EE_LEFT_QUAT = slice(3, 7)
EE_RIGHT_XYZ = slice(7, 10)
EE_RIGHT_QUAT = slice(10, 14)
EE_LEFT_GRIPPER_START = 14
EE_RIGHT_GRIPPER_START = 20
EE_GRIPPER_CHANNELS = 6
EE_RAW_POSE_DIM = 14
EE_DEFAULT_RAW_STATE_DIM = 44
EE_DEFAULT_RAW_ACTION_DIM = 34
EE_RAW_SHARED_STATE_ACTION_DIM = 32
EE_RAW_STATE_WRENCH_SLICE = slice(32, 44)
EE_RAW_ACTION_VELOCITY_SLICE = slice(32, 34)


@dataclass(frozen=True)
class EEActionContract:
    """Validated shared selection contract used by policy, value and stats code."""

    use_rot6d: bool = True
    arm_mode: EEArmMode = "right"
    gripper_dims: int = 1

    def __post_init__(self) -> None:
        if self.arm_mode not in {"left", "right", "both"}:
            raise ValueError(
                f"ee_arm_mode must be one of left/right/both, got {self.arm_mode!r}"
            )
        if not 1 <= int(self.gripper_dims) <= EE_GRIPPER_CHANNELS:
            raise ValueError(
                f"ee_gripper_dims must be in [1, {EE_GRIPPER_CHANNELS}], "
                f"got {self.gripper_dims}"
            )

    @property
    def selected_arms(self) -> tuple[str, ...]:
        if self.arm_mode == "both":
            return ("left", "right")
        return (self.arm_mode,)

    @property
    def pose_dim_per_arm(self) -> int:
        return 9 if self.use_rot6d else 7

    @property
    def physical_dim(self) -> int:
        return len(self.selected_arms) * (self.pose_dim_per_arm + int(self.gripper_dims))

    @property
    def pose_dim(self) -> int:
        return len(self.selected_arms) * self.pose_dim_per_arm

    @property
    def raw_required_dim(self) -> int:
        """Minimum raw EE width needed to read every selected channel."""
        gripper_starts = {
            "left": EE_LEFT_GRIPPER_START,
            "right": EE_RIGHT_GRIPPER_START,
        }
        return max(
            gripper_starts[arm] + int(self.gripper_dims) for arm in self.selected_arms
        )

    @property
    def description(self) -> str:
        rotation = "rot6d(6)" if self.use_rot6d else "quat(4)"
        return (
            f"{self.arm_mode}: xyz(3)+{rotation}+gripper_first({self.gripper_dims}); "
            f"physical_dim={self.physical_dim}"
        )


def make_ee_action_contract(
    *,
    use_rot6d: bool,
    arm_mode: str,
    gripper_dims: int,
) -> EEActionContract:
    return EEActionContract(
        use_rot6d=bool(use_rot6d),
        arm_mode=arm_mode,  # type: ignore[arg-type]
        gripper_dims=int(gripper_dims),
    )


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.as_tensor(quat).to(dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_first_two_cols_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected Rot6D last dim == 6, got {tuple(rot6d.shape)}")
    c1 = rot6d[..., 0:3]
    c2 = rot6d[..., 3:6]
    b1 = F.normalize(c1, dim=-1)
    b2 = F.normalize(c2 - (b1 * c2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion in (x, y, z, w) order."""
    m = matrix.to(dtype=torch.float32)
    batch_shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    quat = torch.zeros(flat.shape[0], 4, dtype=torch.float32, device=m.device)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]

    mask0 = trace > 0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2.0
        quat[mask0, 3] = 0.25 * s
        quat[mask0, 0] = (flat[mask0, 2, 1] - flat[mask0, 1, 2]) / s
        quat[mask0, 1] = (flat[mask0, 0, 2] - flat[mask0, 2, 0]) / s
        quat[mask0, 2] = (flat[mask0, 1, 0] - flat[mask0, 0, 1]) / s

    mask1 = (~mask0) & (flat[:, 0, 0] > flat[:, 1, 1]) & (flat[:, 0, 0] > flat[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1 + flat[mask1, 0, 0] - flat[mask1, 1, 1] - flat[mask1, 2, 2]) * 2
        quat[mask1, 0] = 0.25 * s
        quat[mask1, 1] = (flat[mask1, 0, 1] + flat[mask1, 1, 0]) / s
        quat[mask1, 2] = (flat[mask1, 0, 2] + flat[mask1, 2, 0]) / s
        quat[mask1, 3] = (flat[mask1, 2, 1] - flat[mask1, 1, 2]) / s

    mask2 = (~mask0) & (~mask1) & (flat[:, 1, 1] > flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1 + flat[mask2, 1, 1] - flat[mask2, 0, 0] - flat[mask2, 2, 2]) * 2
        quat[mask2, 0] = (flat[mask2, 0, 1] + flat[mask2, 1, 0]) / s
        quat[mask2, 1] = 0.25 * s
        quat[mask2, 2] = (flat[mask2, 1, 2] + flat[mask2, 2, 1]) / s
        quat[mask2, 3] = (flat[mask2, 0, 2] - flat[mask2, 2, 0]) / s

    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1 + flat[mask3, 2, 2] - flat[mask3, 0, 0] - flat[mask3, 1, 1]) * 2
        quat[mask3, 0] = (flat[mask3, 0, 2] + flat[mask3, 2, 0]) / s
        quat[mask3, 1] = (flat[mask3, 1, 2] + flat[mask3, 2, 1]) / s
        quat[mask3, 2] = 0.25 * s
        quat[mask3, 3] = (flat[mask3, 1, 0] - flat[mask3, 0, 1]) / s

    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return quat.reshape(*batch_shape, 4).to(dtype=matrix.dtype, device=matrix.device)


def _arm_slices(arm: str) -> tuple[slice, slice, int]:
    if arm == "left":
        return EE_LEFT_XYZ, EE_LEFT_QUAT, EE_LEFT_GRIPPER_START
    if arm == "right":
        return EE_RIGHT_XYZ, EE_RIGHT_QUAT, EE_RIGHT_GRIPPER_START
    raise ValueError(f"Unknown arm {arm!r}")


def pack_ee_tensor(x: torch.Tensor, contract: EEActionContract) -> torch.Tensor:
    """Select model-facing EE dimensions from a raw A2D EE tensor."""
    x = torch.as_tensor(x)
    required_dim = max(
        _arm_slices(arm)[2] + contract.gripper_dims for arm in contract.selected_arms
    )
    if x.shape[-1] < required_dim:
        raise ValueError(
            f"{contract.description} requires raw EE last dim >= {required_dim}, "
            f"got {tuple(x.shape)}"
        )
    pose_parts: list[torch.Tensor] = []
    gripper_parts: list[torch.Tensor] = []
    for arm in contract.selected_arms:
        xyz_slice, quat_slice, grip_start = _arm_slices(arm)
        rotation = x[..., quat_slice]
        if contract.use_rot6d:
            rotation = matrix_first_two_cols_to_rot6d(quat_to_matrix(rotation))
        pose_parts.extend([x[..., xyz_slice], rotation])
        gripper_parts.append(x[..., grip_start : grip_start + contract.gripper_dims])
    # Keep all pose channels contiguous so pose-only delta/loss slicing is valid.
    return torch.cat([*pose_parts, *gripper_parts], dim=-1)


def _expand_template_like(template: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    expanded = template
    while expanded.ndim < like.ndim:
        expanded = expanded.unsqueeze(-2)
    return expanded.expand(*like.shape[:-1], expanded.shape[-1])


def unpack_ee_tensor(
    packed: torch.Tensor,
    template: torch.Tensor,
    contract: EEActionContract,
) -> torch.Tensor:
    """Scatter selected model output back into a full raw EE template.

    Unselected arms, unselected gripper channels and robot-specific extras are
    copied from ``template`` (normally the current raw EE state).
    """
    packed = torch.as_tensor(packed)
    if packed.shape[-1] < contract.physical_dim:
        raise ValueError(
            f"Packed action needs {contract.physical_dim} dims for {contract.description}, "
            f"got {tuple(packed.shape)}"
        )
    out = _expand_template_like(
        torch.as_tensor(template, device=packed.device, dtype=packed.dtype), packed
    ).clone()
    pose_offset = 0
    for arm in contract.selected_arms:
        xyz_slice, quat_slice, _ = _arm_slices(arm)
        out[..., xyz_slice] = packed[..., pose_offset : pose_offset + 3]
        pose_offset += 3
        rotation_dim = 6 if contract.use_rot6d else 4
        rotation = packed[..., pose_offset : pose_offset + rotation_dim]
        pose_offset += rotation_dim
        if contract.use_rot6d:
            rotation = matrix_to_quat(rot6d_to_matrix(rotation))
        out[..., quat_slice] = rotation.to(dtype=out.dtype)
    grip_offset = contract.pose_dim
    for arm in contract.selected_arms:
        _, _, grip_start = _arm_slices(arm)
        out[..., grip_start : grip_start + contract.gripper_dims] = packed[
            ..., grip_offset : grip_offset + contract.gripper_dims
        ]
        grip_offset += contract.gripper_dims
    return out


def make_raw_ee_action_template(state: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Build an action template from fields shared with the current raw state.

    The A2D raw layouts share channels 0:32: EE pose/grippers (0:26), head
    joints/commands (26:28), and waist joints/commands (28:32). State channels
    32:44 are wrench measurements, while action channels 32:34 are velocity
    extras with different semantics. The shared prefix is preserved and every
    action-only channel is initialized to zero.
    """
    state = torch.as_tensor(state)
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    shared_dim = min(state.shape[-1], action_dim, EE_RAW_SHARED_STATE_ACTION_DIM)
    return F.pad(state[..., :shared_dim], (0, action_dim - shared_dim))


def pad_or_clip_ee(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        return F.pad(x, (0, dim - x.shape[-1]))
    return x


def mask_padded_ee(x: torch.Tensor, physical_dim: int) -> torch.Tensor:
    """Keep physical EE channels and force model-head padding to zero.

    Flow policies must apply this to both training noise and every inference
    denoising iterate. Otherwise unsupervised padded channels become latent
    random inputs even though the EE contract only defines ``physical_dim``.
    """
    if not 0 < int(physical_dim) <= x.shape[-1]:
        raise ValueError(
            f"physical_dim must be in [1, tensor_dim={x.shape[-1]}], got {physical_dim}"
        )
    if int(physical_dim) == x.shape[-1]:
        return x
    return torch.cat(
        [x[..., : int(physical_dim)], torch.zeros_like(x[..., int(physical_dim) :])],
        dim=-1,
    )
