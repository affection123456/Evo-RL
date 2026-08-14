"""Regression: pi05 Rot6D must attach action chunks to bare or prefixed EE action keys."""

from types import SimpleNamespace

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def _meta(features: dict[str, dict], fps: int = 30):
    return SimpleNamespace(features=features, fps=fps)


def test_resolve_delta_timestamps_pi05_rot6d_bare_ee_actions() -> None:
    cfg = PI05Config(
        use_rot6d=True,
        rot6d_delta_action=True,
        max_state_dim=34,
        max_action_dim=34,
        chunk_size=50,
        ee_action_key="observation.ee_actions",
    )
    meta = _meta(
        {
            "ee_state": {"dtype": "float32", "shape": [44]},
            "ee_actions": {"dtype": "float32", "shape": [34]},
            "actions": {"dtype": "float32", "shape": [34]},
            "top_head": {"dtype": "video", "shape": [720, 1280, 3]},
        }
    )

    delta = resolve_delta_timestamps(cfg, meta)

    assert delta is not None
    assert "ee_actions" in delta
    assert len(delta["ee_actions"]) == cfg.chunk_size
    assert delta["ee_actions"][1] == 1 / meta.fps
    # Joint-space actions must not get a second temporal chunk under Rot6D.
    assert "actions" not in delta
    assert "action" not in delta


def test_resolve_delta_timestamps_pi05_rot6d_prefixed_ee_actions() -> None:
    cfg = PI05Config(
        use_rot6d=True,
        rot6d_delta_action=True,
        max_state_dim=34,
        max_action_dim=34,
        chunk_size=50,
        ee_action_key="observation.ee_actions",
    )
    meta = _meta(
        {
            "observation.ee_state": {"dtype": "float32", "shape": [44]},
            "observation.ee_actions": {"dtype": "float32", "shape": [34]},
            "action": {"dtype": "float32", "shape": [34]},
            "observation.images.top_head": {"dtype": "video", "shape": [720, 1280, 3]},
        }
    )

    delta = resolve_delta_timestamps(cfg, meta)

    assert delta is not None
    assert "observation.ee_actions" in delta
    assert len(delta["observation.ee_actions"]) == cfg.chunk_size
    # Prefer EE quat actions; do not also chunk joint-space `action`.
    assert "action" not in delta
