import torch

from lerobot.policies.ee_action_contract import EEActionContract, pack_ee_tensor, pad_or_clip_ee
from lerobot.policies.pi05.processor_pi05 import (
    Pi05Rot6DDeltaProcessorStep,
    decode_pi05_policy_actions,
    pi05_batch_to_transition,
)
from lerobot.policies.pi0_dmp.processor_pi0_dmp import _quat_pose_to_rot6d
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import NormalizationMode
from lerobot.processor.core import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import OBS_STATE


def test_pi05_right_arm_sets_loss_action_dim_to_physical_width() -> None:
    cfg = PI05Config(use_rot6d=True, ee_arm_mode="right")
    assert cfg.ee_gripper_dims == 1
    assert cfg.loss_action_dim == 10
    cfg6 = PI05Config(use_rot6d=True, ee_arm_mode="right", ee_gripper_dims=6)
    assert cfg6.loss_action_dim == 15


def test_pi05_right_quaternion_contract_has_eight_loss_dims() -> None:
    cfg = PI05Config(use_rot6d=False, ee_arm_mode="right", ee_gripper_dims=1)
    assert cfg.loss_action_dim == 8


def test_pi05_keeps_quantiles_with_rot6d_pad32() -> None:
    cfg = PI05Config(use_rot6d=True)
    assert cfg.max_state_dim == 32
    assert cfg.max_action_dim == 32
    assert cfg.rot6d_delta_action is False
    assert cfg.normalization_mapping["STATE"] == NormalizationMode.QUANTILES
    assert cfg.normalization_mapping["ACTION"] == NormalizationMode.QUANTILES



def test_pi05_batch_to_transition_accepts_dmp_bare_keys() -> None:
    batch = {
        "ee_state": torch.randn(32),
        "ee_actions": torch.randn(32),
        "top_head": torch.randn(3, 8, 8),
        "hand_right": torch.randn(3, 8, 8),
        "hand_left": torch.randn(3, 8, 8),
        "task": ["pick"],
        "index": torch.tensor(0),
    }
    transition = pi05_batch_to_transition(batch)
    obs = transition[TransitionKey.OBSERVATION]
    assert obs is not None
    assert "observation.ee_state" in obs
    assert "observation.ee_actions" in obs
    assert "observation.images.top_head" in obs
    assert "observation.images.hand_right" in obs
    assert "observation.images.hand_left" in obs
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick"]


def test_pi05_batch_to_transition_keeps_basket_observation_keys() -> None:
    batch = {
        "observation.ee_state": torch.randn(32),
        "observation.ee_actions": torch.randn(32),
        "observation.images.top_head": torch.randn(3, 8, 8),
        "observation.state": torch.randn(42),
        "action": torch.randn(32),
        "task": ["basket"],
    }
    transition = pi05_batch_to_transition(batch)
    obs = transition[TransitionKey.OBSERVATION]
    assert obs["observation.ee_state"].shape[-1] == 32
    assert "observation.images.top_head" in obs
    assert transition[TransitionKey.ACTION].shape[-1] == 32


def _normalized_random_pose(batch_size: int, dim: int = 32) -> torch.Tensor:
    pose = torch.randn(batch_size, dim)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    pose[..., 10:14] = torch.nn.functional.normalize(pose[..., 10:14], dim=-1)
    return pose


def test_pi05_rot6d_processor_writes_absolute_by_default() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(4, 32)  # EE quat action chunk

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32)
    transition = create_transition(
        observation={
            "observation.ee_state": state,
            "observation.ee_actions": action,
        },
        # Joint-space distractor must be ignored when EE actions are present.
        action=torch.randn(4, 32),
    )
    out = step(transition)

    obs = out[TransitionKey.OBSERVATION]
    assert OBS_STATE in obs
    assert out[TransitionKey.ACTION].shape == (4, 32)
    assert obs[OBS_STATE].shape[-1] == 32

    contract = EEActionContract(use_rot6d=True, arm_mode="right", gripper_dims=1)
    expected_state = pad_or_clip_ee(pack_ee_tensor(state, contract), 32)
    torch.testing.assert_close(obs[OBS_STATE], expected_state)

    expected_abs = pad_or_clip_ee(pack_ee_tensor(action, contract), 32)
    torch.testing.assert_close(out[TransitionKey.ACTION], expected_abs)


def test_pi05_rot6d_processor_delta_mode() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(4, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=True, arm_mode="both")
    transition = create_transition(
        observation={
            "observation.ee_state": state,
            "observation.ee_actions": action,
        },
        action=torch.randn(4, 32),
    )
    out = step(transition)

    contract = EEActionContract(use_rot6d=True, arm_mode="both", gripper_dims=1)
    expected_state = pad_or_clip_ee(pack_ee_tensor(state, contract), 32)
    abs_action = pad_or_clip_ee(pack_ee_tensor(action, contract), 32)
    expected_delta = abs_action.clone()
    # Pose-only delta (first 18 dims); gripper extras stay absolute.
    expected_delta[..., :18] = abs_action[..., :18] - expected_state[..., :18]
    torch.testing.assert_close(out[TransitionKey.ACTION], expected_delta)


def test_pi05_decode_recovers_quat_action_delta() -> None:
    torch.manual_seed(1)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(3, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=True, arm_mode="both")
    transition = create_transition(
        observation={
            "observation.ee_state": state,
            "observation.ee_actions": action,
        },
        action=torch.randn(3, 32),
    )
    delta = step(transition)[TransitionKey.ACTION]

    class _Cfg:
        use_rot6d = True
        rot6d_delta_action = True
        ee_arm_mode = "both"
        ee_gripper_dims = 1
        max_state_dim = 32
        max_action_dim = 32
        ee_state_key = "observation.ee_state"
        ee_action_key = "observation.ee_actions"

    decoded = decode_pi05_policy_actions(
        delta,
        {"observation.ee_state": state, "observation.ee_actions": action[0]},
        _Cfg(),
    )
    abs_rot6d = _quat_pose_to_rot6d(action)[..., :32]
    decoded_rot6d = _quat_pose_to_rot6d(decoded)[..., :32]
    torch.testing.assert_close(decoded_rot6d[..., :18], abs_rot6d[..., :18], atol=1e-4, rtol=1e-4)


def test_pi05_right_arm_defaults_to_xyz_rot6d_grip1() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 44)[0]
    state[26:32] = torch.arange(6, dtype=state.dtype)
    state[32:44] = 1234.0
    action = _normalized_random_pose(4, 34)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32)
    out = step(
        create_transition(
            observation={"observation.ee_state": state, "observation.ee_actions": action},
            action=torch.randn(4, 34),
        )
    )
    packed = out[TransitionKey.ACTION]
    assert packed.shape == (4, 32)
    # First 10 dims: xyz_r + rot6d_r + grip_r[0]; remainder padded.
    torch.testing.assert_close(packed[..., :3], action[..., 7:10])
    torch.testing.assert_close(packed[..., 9:10], action[..., 20:21])
    assert torch.all(packed[..., 10:] == 0)

    class _Cfg:
        use_rot6d = True
        rot6d_delta_action = False
        ee_arm_mode = "right"
        ee_gripper_dims = 1
        max_state_dim = 32
        max_action_dim = 32
        ee_state_key = "observation.ee_state"
        ee_action_key = "observation.ee_actions"

    decoded = decode_pi05_policy_actions(
        packed,
        {"observation.ee_state": state, "observation.ee_actions": action[0]},
        _Cfg(),
    )
    assert decoded.shape[-1] == 34
    torch.testing.assert_close(decoded[..., :3], state[:3].expand_as(decoded[..., :3]))
    torch.testing.assert_close(decoded[..., 7:10], action[..., 7:10])
    torch.testing.assert_close(decoded[..., 20:21], action[..., 20:21])
    # Unused right-gripper channels stay at the current state.
    torch.testing.assert_close(decoded[..., 21:26], state[21:26].expand_as(decoded[..., 21:26]))
    # Head/waist fields are shared, but state wrench must not leak into action velocity extras.
    torch.testing.assert_close(decoded[..., 26:32], state[26:32].expand_as(decoded[..., 26:32]))
    torch.testing.assert_close(decoded[..., 32:34], torch.zeros_like(decoded[..., 32:34]))


def test_pi05_right_arm_can_keep_all_six_gripper_dims() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 44)[0]
    action = _normalized_random_pose(4, 34)

    step = Pi05Rot6DDeltaProcessorStep(
        state_dim=32, action_dim=32, arm_mode="right", gripper_dims=6
    )
    packed = step(
        create_transition(
            observation={"observation.ee_state": state, "observation.ee_actions": action},
            action=torch.randn(4, 34),
        )
    )[TransitionKey.ACTION]
    torch.testing.assert_close(packed[..., 9:15], action[..., 20:26])
    assert torch.all(packed[..., 15:] == 0)

    class _Cfg:
        use_rot6d = True
        rot6d_delta_action = False
        ee_arm_mode = "right"
        ee_gripper_dims = 6
        max_state_dim = 32
        max_action_dim = 32
        ee_state_key = "observation.ee_state"
        ee_action_key = "observation.ee_actions"

    decoded = decode_pi05_policy_actions(
        packed,
        {"observation.ee_state": state, "observation.ee_actions": action[0]},
        _Cfg(),
    )
    torch.testing.assert_close(decoded[..., 20:26], action[..., 20:26])


def test_pi05_decode_recovers_quat_action_absolute() -> None:
    torch.manual_seed(1)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(3, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=False, arm_mode="both")
    transition = create_transition(
        observation={
            "observation.ee_state": state,
            "observation.ee_actions": action,
        },
        action=torch.randn(3, 32),
    )
    abs_rot6d = step(transition)[TransitionKey.ACTION]

    class _Cfg:
        use_rot6d = True
        rot6d_delta_action = False
        ee_arm_mode = "both"
        ee_gripper_dims = 1
        max_state_dim = 32
        max_action_dim = 32
        ee_state_key = "observation.ee_state"
        ee_action_key = "observation.ee_actions"

    decoded = decode_pi05_policy_actions(
        abs_rot6d,
        {"observation.ee_state": state, "observation.ee_actions": action[0]},
        _Cfg(),
    )
    decoded_rot6d = _quat_pose_to_rot6d(decoded)[..., :32]
    torch.testing.assert_close(decoded_rot6d[..., :18], abs_rot6d[..., :18], atol=1e-4, rtol=1e-4)
