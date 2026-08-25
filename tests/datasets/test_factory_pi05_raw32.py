from types import SimpleNamespace

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def test_pi05_raw32_ee_actions_receive_full_temporal_chunk() -> None:
    cfg = PI05Config(device="cpu", dtype="float32")
    metadata = SimpleNamespace(
        fps=25,
        features={
            "ee_state": {"dtype": "float32", "shape": (44,)},
            "ee_actions": {"dtype": "float32", "shape": (34,)},
            "actions": {"dtype": "float32", "shape": (34,)},
        },
    )

    deltas = resolve_delta_timestamps(
        cfg,
        metadata,
        rename_map={"ee_state": "observation.state", "ee_actions": "action"},
    )

    assert set(deltas) == {"ee_actions"}
    assert len(deltas["ee_actions"]) == 50
    assert deltas["ee_actions"][0] == 0
    assert deltas["ee_actions"][-1] == 49 / 25
