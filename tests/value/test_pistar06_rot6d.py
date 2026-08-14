import torch

from lerobot.policies.pi0_dmp.processor_pi0_dmp import _quat_pose_to_rot6d as policy_quat_pose_to_rot6d
from lerobot.values.pistar06.processor_pistar06 import _quat_pose_to_rot6d as value_quat_pose_to_rot6d


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
