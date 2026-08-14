import torch

from lerobot.configs.types import NormalizationMode
from lerobot.policies.ee_action_contract import EEActionContract, pack_ee_tensor, pad_or_clip_ee
from lerobot.policies.pi0_dmp.configuration_pi0_dmp import PI0DMPConfig
from lerobot.policies.pi0_dmp.processor_pi0_dmp import (
    PI0DMPRot6DDeltaProcessorStep,
    _quat_pose_to_rot6d,
    _rot6d_pose_to_quat,
    decode_pi0_dmp_policy_actions,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_STATE


def _normalized_random_pose(batch_size: int, dim: int = 30) -> torch.Tensor:
    pose = torch.randn(batch_size, dim)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    pose[..., 10:14] = torch.nn.functional.normalize(pose[..., 10:14], dim=-1)
    return pose


def test_pi0_dmp_defaults_match_pi05_lerobotv3_norm() -> None:
    cfg = PI0DMPConfig()
    assert cfg.max_state_dim == 32
    assert cfg.max_action_dim == 32
    assert cfg.rot6d_delta_action is False
    assert cfg.ee_arm_mode == "right"
    assert cfg.ee_gripper_dims == 1
    assert cfg.loss_action_dim == 10
    assert cfg.normalization_mapping["STATE"] == NormalizationMode.QUANTILES
    assert cfg.normalization_mapping["ACTION"] == NormalizationMode.QUANTILES


def test_pi0_dmp_processor_writes_absolute_by_default() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 32)[0]
    action = _normalized_random_pose(4, 32)
    ref_state = _normalized_random_pose(1, 32)[0]
    ref_action = _normalized_random_pose(4, 32)

    step = PI0DMPRot6DDeltaProcessorStep()
    out = step(
        create_transition(
            observation={
                OBS_STATE: state,
                "observation.reference.state": ref_state,
                "observation.ref_actions": ref_action,
            },
            action=action,
        )
    )
    contract = EEActionContract(use_rot6d=True, arm_mode="right", gripper_dims=1)
    expected_action = pad_or_clip_ee(pack_ee_tensor(action, contract), 32)
    expected_ref = pad_or_clip_ee(pack_ee_tensor(ref_action, contract), 32)
    torch.testing.assert_close(out[TransitionKey.ACTION], expected_action)
    torch.testing.assert_close(
        out[TransitionKey.OBSERVATION]["observation.ref_actions"], expected_ref
    )
    assert out[TransitionKey.OBSERVATION][OBS_STATE].shape[-1] == 32


def test_pi0_dmp_decode_preserves_shared_fields_without_copying_state_wrench() -> None:
    state = _normalized_random_pose(1, 44)[0]
    state[26:32] = torch.arange(6, dtype=state.dtype)
    state[32:44] = 1234.0
    action = _normalized_random_pose(1, 34)
    contract = EEActionContract(use_rot6d=True, arm_mode="right", gripper_dims=1)
    packed = pad_or_clip_ee(pack_ee_tensor(action, contract), 32)

    decoded = decode_pi0_dmp_policy_actions(
        packed,
        {OBS_STATE: state, "ee_actions": action[0]},
        PI0DMPConfig(),
    )

    torch.testing.assert_close(decoded[..., 7:10], action[..., 7:10])
    torch.testing.assert_close(decoded[..., 20:21], action[..., 20:21])
    torch.testing.assert_close(decoded[..., 26:32], state[26:32].expand_as(decoded[..., 26:32]))
    torch.testing.assert_close(decoded[..., 32:34], torch.zeros_like(decoded[..., 32:34]))


def _quat_error_deg(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    dot = (actual * expected).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot) * 180.0 / torch.pi


def test_rot6d_quaternion_round_trip() -> None:
    torch.manual_seed(0)
    pose = _normalized_random_pose(256)

    decoded = _rot6d_pose_to_quat(_quat_pose_to_rot6d(pose))

    torch.testing.assert_close(decoded[..., :3], pose[..., :3])
    torch.testing.assert_close(decoded[..., 7:10], pose[..., 7:10])
    torch.testing.assert_close(decoded[..., 14:], pose[..., 14:])
    assert _quat_error_deg(decoded[..., 3:7], pose[..., 3:7]).max() < 0.1
    assert _quat_error_deg(decoded[..., 10:14], pose[..., 10:14]).max() < 0.1


def test_rot6d_identity_quaternion_round_trip() -> None:
    pose = torch.zeros(1, 30)
    pose[..., 6] = 1.0
    pose[..., 13] = 1.0

    decoded = _rot6d_pose_to_quat(_quat_pose_to_rot6d(pose))

    torch.testing.assert_close(decoded, pose)
