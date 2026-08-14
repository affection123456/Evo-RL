#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pi0.configuration_pi0 import PI0Config


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

    # Historical full32 dual-arm Rot6D representation.
    use_rot6d: bool = True
    ee_raw_action_dim: int = 34

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
        if not self.use_rot6d:
            raise ValueError("PI0-DMP full32 processor requires use_rot6d=True")
        if self.max_state_dim < 18:
            raise ValueError(
                f"use_rot6d requires max_state_dim >= 18 for dual-arm xyz+rot6d, got {self.max_state_dim}"
            )
        if self.max_action_dim < 18:
            raise ValueError(
                f"use_rot6d requires max_action_dim >= 18 for dual-arm xyz+rot6d, got {self.max_action_dim}"
            )
