#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.ee_action_contract import make_ee_action_contract


@PreTrainedConfig.register_subclass("pi0_dmp")
@dataclass
class PI0DMPConfig(PI0Config):
    """PI0-DMP config with reference trajectory support.

    Norm/pad defaults match verified pi05_data_lerobotv3:
    QUANTILES + pad32 + absolute Rot6D (no delta).
    """

    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 50
    n_action_steps: int = 50

    # Shared model-facing EE contract.
    use_rot6d: bool = True
    ee_arm_mode: str = "right"  # left | right | both
    ee_gripper_dims: int = 1
    ee_raw_action_dim: int = 34
    loss_action_dim: int | None = None

    # Absolute Rot6D by default; set True for pose-only delta vs state
    # (requires matching delta action / ref_actions stats).
    rot6d_delta_action: bool = False

    # Additional keys used by PI0-DMP training.
    ref_state_key: str = "observation.reference.state"
    ref_action_key: str = "observation.ref_actions"
    ref_image_features: tuple[str, str] = (
        "observation.images.ref_top_head",
        "observation.images.ref_hand_right",
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Match the OpenPI DMP training preset.
    optimizer_lr: float = 5.0e-5
    scheduler_decay_lr: float = 6.0e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        contract = make_ee_action_contract(
            use_rot6d=self.use_rot6d,
            arm_mode=self.ee_arm_mode,
            gripper_dims=self.ee_gripper_dims,
        )
        if self.loss_action_dim is None:
            object.__setattr__(self, "loss_action_dim", contract.physical_dim)
        if self.max_state_dim < contract.physical_dim:
            raise ValueError(
                f"max_state_dim={self.max_state_dim} is smaller than {contract.description}"
            )
        if self.max_action_dim < contract.physical_dim:
            raise ValueError(
                f"max_action_dim={self.max_action_dim} is smaller than {contract.description}"
            )
        if not 0 < int(self.loss_action_dim) <= contract.physical_dim:
            raise ValueError(
                f"loss_action_dim must be in [1, physical_dim={contract.physical_dim}], "
                f"got {self.loss_action_dim}"
            )
