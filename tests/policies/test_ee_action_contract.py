import torch

from lerobot.policies import ee_action_contract as contract_module
from lerobot.policies.ee_action_contract import (
    EEActionContract,
    mask_padded_ee,
    pack_ee_tensor,
    unpack_ee_tensor,
)


def _raw_ee(shape: tuple[int, ...] = (2, 34)) -> torch.Tensor:
    raw = torch.randn(*shape)
    raw[..., 3:7] = torch.nn.functional.normalize(raw[..., 3:7], dim=-1)
    raw[..., 10:14] = torch.nn.functional.normalize(raw[..., 10:14], dim=-1)
    return raw


def test_right_first_gripper_physical_dimensions() -> None:
    assert EEActionContract(use_rot6d=False).physical_dim == 8
    assert EEActionContract(use_rot6d=True).physical_dim == 10


def test_right_quaternion_contract_selects_xyz_quat_gripper_first() -> None:
    raw = _raw_ee()
    contract = EEActionContract(use_rot6d=False, arm_mode="right", gripper_dims=1)
    packed = pack_ee_tensor(raw, contract)

    assert packed.shape[-1] == 8
    torch.testing.assert_close(packed[..., :3], raw[..., 7:10])
    torch.testing.assert_close(packed[..., 3:7], raw[..., 10:14])
    torch.testing.assert_close(packed[..., 7:8], raw[..., 20:21])


def test_right_contract_scatter_preserves_unselected_raw_channels() -> None:
    state = _raw_ee((34,))
    target = _raw_ee((4, 34))
    for use_rot6d in (False, True):
        contract = EEActionContract(
            use_rot6d=use_rot6d,
            arm_mode="right",
            gripper_dims=1,
        )
        decoded = unpack_ee_tensor(pack_ee_tensor(target, contract), state, contract)

        torch.testing.assert_close(decoded[..., 7:10], target[..., 7:10])
        torch.testing.assert_close(decoded[..., 20:21], target[..., 20:21])
        torch.testing.assert_close(decoded[..., :3], state[:3].expand_as(decoded[..., :3]))
        torch.testing.assert_close(decoded[..., 14:20], state[14:20].expand_as(decoded[..., 14:20]))
        torch.testing.assert_close(decoded[..., 21:], state[21:].expand_as(decoded[..., 21:]))


def test_raw_action_template_preserves_shared_state_fields_and_zeros_action_extras() -> None:
    state = _raw_ee((44,))
    state[26:32] = torch.arange(6, dtype=state.dtype)
    state[32:44] = 1234.0

    template = contract_module.make_raw_ee_action_template(state, action_dim=34)

    torch.testing.assert_close(template[:32], state[:32])
    torch.testing.assert_close(template[32:34], torch.zeros(2, dtype=state.dtype))


def test_both_arm_contract_keeps_pose_before_grippers() -> None:
    raw = _raw_ee()
    contract = EEActionContract(use_rot6d=False, arm_mode="both", gripper_dims=1)
    packed = pack_ee_tensor(raw, contract)

    assert contract.pose_dim == 14
    assert contract.physical_dim == 16
    torch.testing.assert_close(packed[..., :7], raw[..., :7])
    torch.testing.assert_close(packed[..., 7:14], raw[..., 7:14])
    torch.testing.assert_close(packed[..., 14:15], raw[..., 14:15])
    torch.testing.assert_close(packed[..., 15:16], raw[..., 20:21])


def test_mask_padded_ee_keeps_physical_channels_and_zeros_padding() -> None:
    value = torch.randn(2, 3, 32)
    masked = mask_padded_ee(value, physical_dim=10)

    torch.testing.assert_close(masked[..., :10], value[..., :10])
    torch.testing.assert_close(masked[..., 10:], torch.zeros_like(masked[..., 10:]))


def test_mask_padded_ee_preserves_gradient_on_physical_channels_only() -> None:
    value = torch.randn(2, 32, requires_grad=True)
    mask_padded_ee(value, physical_dim=10).sum().backward()

    torch.testing.assert_close(value.grad[..., :10], torch.ones_like(value.grad[..., :10]))
    torch.testing.assert_close(value.grad[..., 10:], torch.zeros_like(value.grad[..., 10:]))
