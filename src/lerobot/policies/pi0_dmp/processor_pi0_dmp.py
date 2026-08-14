#!/usr/bin/env python

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.ee_action_contract import (
    EEActionContract,
    make_ee_action_contract,
    make_raw_ee_action_template,
    pack_ee_tensor,
    pad_or_clip_ee,
    unpack_ee_tensor,
)
from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
from lerobot.policies.pi0_dmp.configuration_pi0_dmp import PI0DMPConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


OBS_TOP_HEAD = "observation.images.top_head"
OBS_HAND_RIGHT = "observation.images.hand_right"
OBS_REF_TOP_HEAD = "observation.images.ref_top_head"
OBS_REF_HAND_RIGHT = "observation.images.ref_hand_right"
OBS_REF_STATE = "observation.reference.state"
OBS_REF_ACTIONS = "observation.ref_actions"
# Dual-arm xyz + Rot6D [r1,r2]; gripper extras stay absolute when delta is on.
PI0_DMP_ROT6D_POSE_DIM = 18


def _get_first(batch: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    raise KeyError(f"Missing any of keys: {keys}")


def pi0_dmp_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Pack raw DMP LeRobot fields into canonical observation/action keys.

    This intentionally supports both the raw openpi-style DMP keys and already
    canonical LeRobot policy keys.
    """
    observation = {
        OBS_STATE: _get_first(batch, OBS_STATE, "ee_state"),
        OBS_REF_STATE: _get_first(batch, OBS_REF_STATE, "ref_ee_state"),
        OBS_REF_ACTIONS: _get_first(batch, OBS_REF_ACTIONS, "ref_ee_actions"),
        OBS_TOP_HEAD: _get_first(batch, OBS_TOP_HEAD, "top_head"),
        OBS_HAND_RIGHT: _get_first(batch, OBS_HAND_RIGHT, "hand_right"),
        OBS_REF_TOP_HEAD: _get_first(batch, OBS_REF_TOP_HEAD, "ref_top_head"),
        OBS_REF_HAND_RIGHT: _get_first(batch, OBS_REF_HAND_RIGHT, "ref_hand_right"),
    }
    if ACTION in batch or "ee_actions" in batch:
        action = _get_first(batch, ACTION, "ee_actions")
    else:
        action = None
    complementary_data = {
        key: batch[key]
        for key in ("task", "index", "task_index", "episode_index", "frame_index")
        if key in batch
    }
    return create_transition(observation=observation, action=action, complementary_data=complementary_data)


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _matrix_first_two_cols_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the first two rotation-matrix columns as ``[col0, col1]``."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _quat_pose_to_rot6d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 14:
        raise ValueError(f"Expected last dim >= 14 for dual-arm xyz+quat layout, got {tuple(x.shape)}")
    left = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 3:7]))
    right = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 10:14]))
    return torch.cat([x[..., :3], left, x[..., 7:10], right, x[..., 14:]], dim=-1)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from a 6D representation (first two columns)."""
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected Rot6D last dim == 6, got {tuple(rot6d.shape)}")
    c1 = rot6d[..., 0:3]
    c2 = rot6d[..., 3:6]
    b1 = F.normalize(c1, dim=-1)
    proj = (b1 * c2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(c2 - proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion in (x, y, z, w) order (matches ``_quat_to_matrix``)."""
    m = matrix.to(dtype=torch.float32)
    batch_shape = m.shape[:-2]
    m_flat = m.reshape(-1, 3, 3)

    trace = m_flat[:, 0, 0] + m_flat[:, 1, 1] + m_flat[:, 2, 2]
    quat = torch.zeros(m_flat.shape[0], 4, dtype=torch.float32, device=m.device)

    mask0 = trace > 0.0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2.0
        quat[mask0, 3] = 0.25 * s
        quat[mask0, 0] = (m_flat[mask0, 2, 1] - m_flat[mask0, 1, 2]) / s
        quat[mask0, 1] = (m_flat[mask0, 0, 2] - m_flat[mask0, 2, 0]) / s
        quat[mask0, 2] = (m_flat[mask0, 1, 0] - m_flat[mask0, 0, 1]) / s

    mask1 = (~mask0) & (m_flat[:, 0, 0] > m_flat[:, 1, 1]) & (m_flat[:, 0, 0] > m_flat[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1.0 + m_flat[mask1, 0, 0] - m_flat[mask1, 1, 1] - m_flat[mask1, 2, 2]) * 2.0
        quat[mask1, 0] = 0.25 * s
        quat[mask1, 1] = (m_flat[mask1, 0, 1] + m_flat[mask1, 1, 0]) / s
        quat[mask1, 2] = (m_flat[mask1, 0, 2] + m_flat[mask1, 2, 0]) / s
        quat[mask1, 3] = (m_flat[mask1, 2, 1] - m_flat[mask1, 1, 2]) / s

    mask2 = (~mask0) & (~mask1) & (m_flat[:, 1, 1] > m_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + m_flat[mask2, 1, 1] - m_flat[mask2, 0, 0] - m_flat[mask2, 2, 2]) * 2.0
        quat[mask2, 0] = (m_flat[mask2, 0, 1] + m_flat[mask2, 1, 0]) / s
        quat[mask2, 1] = 0.25 * s
        quat[mask2, 2] = (m_flat[mask2, 1, 2] + m_flat[mask2, 2, 1]) / s
        quat[mask2, 3] = (m_flat[mask2, 0, 2] - m_flat[mask2, 2, 0]) / s

    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + m_flat[mask3, 2, 2] - m_flat[mask3, 0, 0] - m_flat[mask3, 1, 1]) * 2.0
        quat[mask3, 0] = (m_flat[mask3, 0, 2] + m_flat[mask3, 2, 0]) / s
        quat[mask3, 1] = (m_flat[mask3, 1, 2] + m_flat[mask3, 2, 1]) / s
        quat[mask3, 2] = 0.25 * s
        quat[mask3, 3] = (m_flat[mask3, 1, 0] - m_flat[mask3, 0, 1]) / s

    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return quat.reshape(*batch_shape, 4).to(dtype=matrix.dtype, device=matrix.device)


def _rot6d_pose_to_quat(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_quat_pose_to_rot6d`` for dual-arm xyz+rot6d layouts."""
    if x.shape[-1] < 18:
        raise ValueError(f"Expected last dim >= 18 for dual-arm xyz+rot6d layout, got {tuple(x.shape)}")
    left_quat = _matrix_to_quat(_rot6d_to_matrix(x[..., 3:9]))
    right_quat = _matrix_to_quat(_rot6d_to_matrix(x[..., 12:18]))
    return torch.cat([x[..., :3], left_quat, x[..., 9:12], right_quat, x[..., 18:]], dim=-1)


def _resolve_quat_action_dim(raw_observation: dict[str, Any], policy_cfg: PI0DMPConfig | Any) -> int | None:
    for key in ("ee_actions", ACTION, "action"):
        if key in raw_observation:
            value = torch.as_tensor(raw_observation[key])
            return int(value.shape[-1])
    raw_action_dim = getattr(policy_cfg, "ee_raw_action_dim", None)
    if raw_action_dim is not None:
        return int(raw_action_dim)
    output_features = getattr(policy_cfg, "output_features", None) or {}
    action_feature = output_features.get(ACTION)
    if action_feature is not None and getattr(action_feature, "shape", None):
        return int(action_feature.shape[0])
    return None


def decode_pi0_dmp_policy_actions(
    actions: torch.Tensor,
    raw_observation: dict[str, Any],
    policy_cfg: PI0DMPConfig | Any,
) -> torch.Tensor:
    """Decode selected model output and scatter it into a full raw EE action."""
    state_value = None
    for key in (OBS_STATE, "observation/state", "ee_state"):
        if key in raw_observation:
            state_value = raw_observation[key]
            break
    if state_value is None:
        raise KeyError("PI0-DMP EE contract decode needs raw EE state as a full-action template.")

    max_state_dim = getattr(policy_cfg, "max_state_dim", actions.shape[-1])
    max_action_dim = getattr(policy_cfg, "max_action_dim", actions.shape[-1])
    use_delta = bool(getattr(policy_cfg, "rot6d_delta_action", False))
    contract = make_ee_action_contract(
        use_rot6d=bool(getattr(policy_cfg, "use_rot6d", True)),
        arm_mode=getattr(policy_cfg, "ee_arm_mode", "right"),
        gripper_dims=int(getattr(policy_cfg, "ee_gripper_dims", 1)),
    )

    state = pad_or_clip_ee(
        pack_ee_tensor(torch.as_tensor(state_value), contract), max_state_dim
    )
    if state.ndim == 1:
        state = state.unsqueeze(0)
    state = state.to(device=actions.device, dtype=actions.dtype)

    decoded = actions.clone()
    if use_delta:
        delta_dims = min(contract.pose_dim, max_action_dim, max_state_dim)
        if delta_dims > 0:
            state_for_delta = state
            if decoded.ndim == state.ndim + 1:
                state_for_delta = state.unsqueeze(-2)
            decoded[..., :delta_dims] = decoded[..., :delta_dims] + state_for_delta[..., :delta_dims]

    target_dim = _resolve_quat_action_dim(raw_observation, policy_cfg)
    if target_dim is None:
        target_dim = int(torch.as_tensor(state_value).shape[-1])
    template = make_raw_ee_action_template(torch.as_tensor(state_value), target_dim)
    return unpack_ee_tensor(decoded, template, contract)


def _pad_or_clip(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        return F.pad(x, (0, dim - x.shape[-1]))
    return x


@ProcessorStepRegistry.register(name="pi0_dmp_rot6d_delta_processor")
class PI0DMPRot6DDeltaProcessorStep(ProcessorStep):
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 32,
        state_key: str = OBS_STATE,
        ref_state_key: str = OBS_REF_STATE,
        ref_action_key: str = OBS_REF_ACTIONS,
        use_delta: bool = False,
        delta_dims: int | None = None,
        use_rot6d: bool = True,
        arm_mode: str = "right",
        gripper_dims: int = 1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_key = state_key
        self.ref_state_key = ref_state_key
        self.ref_action_key = ref_action_key
        self.use_delta = use_delta
        self.use_rot6d = use_rot6d
        self.arm_mode = arm_mode
        self.gripper_dims = gripper_dims
        self.delta_dims = delta_dims

    def _contract(self) -> EEActionContract:
        return make_ee_action_contract(
            use_rot6d=self.use_rot6d,
            arm_mode=self.arm_mode,
            gripper_dims=self.gripper_dims,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        contract = self._contract()
        state = pad_or_clip_ee(
            pack_ee_tensor(torch.as_tensor(observation[self.state_key]), contract), self.state_dim
        )
        ref_state = pad_or_clip_ee(
            pack_ee_tensor(torch.as_tensor(observation[self.ref_state_key]), contract), self.state_dim
        )
        raw_action = transition.get(TransitionKey.ACTION)
        actions = (
            None
            if raw_action is None
            else pad_or_clip_ee(pack_ee_tensor(torch.as_tensor(raw_action), contract), self.action_dim)
        )
        ref_actions = pad_or_clip_ee(
            pack_ee_tensor(torch.as_tensor(observation[self.ref_action_key]), contract), self.action_dim
        )

        if self.use_delta:
            delta_dims = contract.pose_dim if self.delta_dims is None else self.delta_dims
            dims = min(delta_dims, self.action_dim, self.state_dim)
            if dims > 0:
                ref_actions = ref_actions.clone()
                if actions is not None:
                    actions = actions.clone()
                    actions[..., :dims] = actions[..., :dims] - state[..., :dims].unsqueeze(-2)
                ref_actions[..., :dims] = ref_actions[..., :dims] - ref_state[..., :dims].unsqueeze(-2)

        observation[self.state_key] = state
        observation[self.ref_state_key] = ref_state
        observation[self.ref_action_key] = ref_actions
        transition[TransitionKey.OBSERVATION] = observation
        if actions is not None:
            transition[TransitionKey.ACTION] = actions
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_key": self.state_key,
            "ref_state_key": self.ref_state_key,
            "ref_action_key": self.ref_action_key,
            "use_delta": self.use_delta,
            "delta_dims": self.delta_dims,
            "use_rot6d": self.use_rot6d,
            "arm_mode": self.arm_mode,
            "gripper_dims": self.gripper_dims,
        }


def make_pi0_dmp_pre_post_processors(
    config: PI0DMPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps: list[ProcessorStep] = [
        PI0DMPRot6DDeltaProcessorStep(
            state_dim=config.max_state_dim,
            action_dim=config.max_action_dim,
            state_key=OBS_STATE,
            ref_state_key=config.ref_state_key,
            ref_action_key=config.ref_action_key,
            use_delta=bool(getattr(config, "rot6d_delta_action", False)),
            use_rot6d=bool(getattr(config, "use_rot6d", True)),
            arm_mode=getattr(config, "ee_arm_mode", "right"),
            gripper_dims=int(getattr(config, "ee_gripper_dims", 1)),
        ),
        AddBatchDimensionProcessorStep(),
        Pi0NewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=pi0_dmp_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
