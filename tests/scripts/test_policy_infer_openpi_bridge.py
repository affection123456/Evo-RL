from types import SimpleNamespace

import numpy as np

from lerobot.scripts.lerobot_policy_infer_openpi_bridge import LeRobotOpenPIBridge


def test_bridge_accepts_full32_dual_arm_raw_state_on_first_request() -> None:
    bridge = LeRobotOpenPIBridge.__new__(LeRobotOpenPIBridge)
    bridge._policy_cfg = SimpleNamespace(
        type="pi05",
        ee_state_key="observation.ee_state",
        image_features={},
    )
    bridge._logged_observation_schema = False
    observation = {
        "observation.ee_state": np.zeros(14, dtype=np.float32),
        "observation.images.top_head": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.hand_right": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    bridge._validate_and_log_observation(observation)

    assert bridge._logged_observation_schema
