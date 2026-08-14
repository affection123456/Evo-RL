import pytest
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
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


def test_pi05_flow_noise_covers_all_32_action_dimensions() -> None:
    model = PI05Pytorch.__new__(PI05Pytorch)
    torch.nn.Module.__init__(model)
    model.config = type("_Cfg", (), {})()
    torch.manual_seed(0)
    noise = model.sample_noise((2, 3, 32), torch.device("cpu"))
    assert torch.count_nonzero(noise[..., 10:]) > 0


def test_pi05_keeps_quantiles_with_rot6d_pad32() -> None:
    cfg = PI05Config(use_rot6d=True)
    assert cfg.max_state_dim == 32
    assert cfg.max_action_dim == 32
    assert cfg.rot6d_delta_action is False
    assert cfg.normalization_mapping["STATE"] == NormalizationMode.QUANTILES
    assert cfg.normalization_mapping["ACTION"] == NormalizationMode.QUANTILES


def test_pi05_rot6d_requires_dual_arm_pose_width() -> None:
    with pytest.raises(ValueError, match="max_action_dim >= 18"):
        PI05Config(use_rot6d=True, max_action_dim=17)



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

    expected_state = _quat_pose_to_rot6d(state)[..., :32]
    torch.testing.assert_close(obs[OBS_STATE], expected_state)

    expected_abs = _quat_pose_to_rot6d(action)[..., :32]
    torch.testing.assert_close(out[TransitionKey.ACTION], expected_abs)


def test_pi05_rot6d_processor_delta_mode() -> None:
    torch.manual_seed(0)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(4, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=True)
    transition = create_transition(
        observation={
            "observation.ee_state": state,
            "observation.ee_actions": action,
        },
        action=torch.randn(4, 32),
    )
    out = step(transition)

    expected_state = _quat_pose_to_rot6d(state)[..., :32]
    abs_action = _quat_pose_to_rot6d(action)[..., :32]
    expected_delta = abs_action.clone()
    # Pose-only delta (first 18 dims); gripper extras stay absolute.
    expected_delta[..., :18] = abs_action[..., :18] - expected_state[..., :18]
    torch.testing.assert_close(out[TransitionKey.ACTION], expected_delta)


def test_pi05_decode_recovers_quat_action_delta() -> None:
    torch.manual_seed(1)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(3, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=True)
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


def test_pi05_decode_recovers_quat_action_absolute() -> None:
    torch.manual_seed(1)
    state = _normalized_random_pose(1, 42)[0]
    action = _normalized_random_pose(3, 32)

    step = Pi05Rot6DDeltaProcessorStep(state_dim=32, action_dim=32, use_delta=False)
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
