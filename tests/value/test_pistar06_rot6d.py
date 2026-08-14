import pytest
import torch

from lerobot.policies.pi0_dmp.processor_pi0_dmp import _quat_pose_to_rot6d as policy_quat_pose_to_rot6d
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.processor_pistar06 import (
    Pistar06Rot6DStateProcessorStep,
    _quat_pose_to_rot6d as value_quat_pose_to_rot6d,
)


def test_pistar06_rot6d_matches_pi0_dmp_col_major_layout() -> None:
    torch.manual_seed(0)
    pose = torch.randn(8, 30)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    pose[..., 10:14] = torch.nn.functional.normalize(pose[..., 10:14], dim=-1)

    torch.testing.assert_close(value_quat_pose_to_rot6d(pose), policy_quat_pose_to_rot6d(pose))


def test_pistar06_rot6d_uses_col0_col1_layout() -> None:
    # 90-deg about z: R = [[0,-1,0],[1,0,0],[0,0,1]] -> quat (x,y,z,w) = (0,0,√2/2,√2/2)
    # [col0, col1] = [0,1,0, -1,0,0]
    s = 0.5**0.5
    pose = torch.zeros(1, 16)
    pose[..., 3:7] = torch.tensor([0.0, 0.0, s, s])
    pose[..., 10:14] = torch.tensor([0.0, 0.0, s, s])

    rot6d = value_quat_pose_to_rot6d(pose)
    expected = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    torch.testing.assert_close(rot6d[..., 3:9], expected)
    torch.testing.assert_close(rot6d[..., 12:18], expected)


def test_pistar06_processor_keeps_dual_arm_pose_and_raw_tail() -> None:
    pose = torch.zeros(30)
    pose[6] = 1.0
    pose[13] = 1.0
    pose[14:] = torch.arange(16, dtype=torch.float32)
    step = Pistar06Rot6DStateProcessorStep(max_state_dim=32)
    out = step(create_transition(observation={"observation.state": pose}))
    actual = out[TransitionKey.OBSERVATION]["observation.state"]
    expected = value_quat_pose_to_rot6d(pose)[..., :32]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[18:], pose[14:28])


def test_pistar06_rot6d_requires_dual_arm_pose_width() -> None:
    with pytest.raises(ValueError, match="max_state_dim.*18"):
        Pistar06Config(use_rot6d=True, max_state_dim=17)
