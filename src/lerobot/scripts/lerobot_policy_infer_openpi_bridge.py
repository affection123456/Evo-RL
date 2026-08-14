#!/usr/bin/env python

"""Serve a LeRobot checkpoint over the OpenPI msgpack WebSocket protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import numpy as np
import torch
from typing_extensions import override

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.lerobot_policy_infer import (
    PolicyInferWebsocketConfig,
    WebsocketServerConfig,
    _PolicySession,
    _clone_cache,
    _resolve_pretrained_model_path,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from openpi.serving import websocket_policy_server
from openpi_client import base_policy as _base_policy
import websockets.asyncio.server as _ws_server

# Robot trajectories can run for several minutes between inferences; default
# websockets keepalive (~20s) is too aggressive for a long-lived connection.
DEFAULT_PING_INTERVAL_S = 30
DEFAULT_PING_TIMEOUT_S = 600


def _pi0_dmp_denoise_step_cached(
    model: Any,
    *,
    state: torch.Tensor,
    ref_state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values: Any,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Run one PI0-DMP suffix step against a cached vision/language prefix."""
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
        state, x_t, timestep, ref_state=ref_state
    )
    if (
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        # Transformers DynamicCache is mutated even with use_cache=False.
        # Every flow step must start from the same prefix-only cache.
        past_key_values=_clone_cache(past_key_values),
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


@torch.no_grad()
def _pi0_dmp_sample_actions_cached(
    model: Any,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    ref_state: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    num_steps: int | None = None,
) -> torch.Tensor:
    """Sample PI0-DMP actions while computing the image/language prefix once."""
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    if num_steps is None:
        num_steps = model.config.num_inference_steps
    if ref_state is None:
        ref_state = state

    batch_size = state.shape[0]
    device = state.device
    if noise is None:
        noise = model.sample_noise(
            (batch_size, model.config.chunk_size, model.config.max_action_dim),
            device,
        )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    if (
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)
        velocity = _pi0_dmp_denoise_step_cached(
            model,
            state=state,
            ref_state=ref_state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )
        x_t = x_t + dt * velocity
    return x_t


def _enable_pi0_dmp_kv_cache(policy: Any) -> None:
    model = getattr(policy, "model", None)
    if model is None or model.__class__.__name__ != "PI0DMPPytorch":
        raise TypeError(
            "PI0-DMP KV cache requires policy.model to be PI0DMPPytorch, "
            f"got {type(model).__name__}."
        )
    model.sample_actions = MethodType(_pi0_dmp_sample_actions_cached, model)
    logging.info("Enabled PI0-DMP inference KV cache (vision/language prefix computed once per request).")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unwrap_client_payload(obs: dict[str, Any]) -> dict[str, Any]:
    if "obs" in obs and isinstance(obs["obs"], dict):
        return obs["obs"]
    return obs


def _first_present(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _openpi_obs_to_lerobot(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normalize OpenPI client payloads to the flat keys used by ``rename_map``.

    Supports both PI0-DMP (current + reference cameras) and PI05 (current cameras
    only, optionally including left wrist).
    """
    payload = _unwrap_client_payload(obs)
    out: dict[str, np.ndarray] = {}

    def put(dst: str, value: Any) -> None:
        if value is None:
            return
        out[dst] = _as_numpy(value)

    put("observation/state", _first_present(payload, "observation/state", "state"))
    put("reference/state", _first_present(payload, "reference/state", "ref_state"))
    put("ref_actions", _first_present(payload, "ref_actions", "ref_ee_actions"))

    images = payload.get("images")
    if isinstance(images, dict):
        put("observation/image", _first_present(images, "top_head", "observation/image"))
        put(
            "observation/right_wrist_image",
            _first_present(images, "hand_right", "right_wrist_image", "observation/right_wrist_image"),
        )
        put(
            "observation/left_wrist_image",
            _first_present(images, "hand_left", "left_wrist_image", "observation/left_wrist_image"),
        )
        put("reference/image", _first_present(images, "ref_top_head", "reference/image"))
        put(
            "reference/right_wrist_image",
            _first_present(images, "ref_hand_right", "reference/right_wrist_image"),
        )
    else:
        put("observation/image", payload.get("observation/image"))
        put("observation/right_wrist_image", payload.get("observation/right_wrist_image"))
        put("observation/left_wrist_image", payload.get("observation/left_wrist_image"))
        put("reference/image", payload.get("reference/image"))
        put("reference/right_wrist_image", payload.get("reference/right_wrist_image"))

    return out


def _required_policy_observation_keys(policy_cfg: PreTrainedConfig) -> set[str]:
    """Required keys after ``rename_map`` for OpenPI bridge validation."""
    policy_type = getattr(policy_cfg, "type", None)
    if policy_type == "pi0_dmp":
        return {
            "observation.state",
            "observation.reference.state",
            "observation.ref_actions",
            "observation.images.top_head",
            "observation.images.hand_right",
            "observation.images.ref_top_head",
            "observation.images.ref_hand_right",
        }
    if policy_type == "pi05":
        # PI05 Rot6D reads EE quat from ee_state_key; images come from image_features.
        required = {getattr(policy_cfg, "ee_state_key", "observation.ee_state") or "observation.ee_state"}
        image_features = getattr(policy_cfg, "image_features", None) or {}
        # Always require the two primary robot cameras; left wrist is optional at
        # runtime (missing cameras are zero-masked inside PI05).
        required.update({"observation.images.top_head", "observation.images.hand_right"})
        for key in image_features:
            if key.endswith(".hand_left"):
                # Prefer left wrist when the checkpoint was trained with it, but do
                # not hard-fail if the robot client cannot provide it yet.
                continue
            if str(key).startswith("observation.images."):
                required.add(str(key))
        return required
    raise ValueError(
        f"OpenPI bridge validation is not configured for policy.type={policy_type!r}. "
        "Supported types: pi0_dmp, pi05."
    )


def _response_state_key(policy_cfg: PreTrainedConfig) -> str:
    if getattr(policy_cfg, "type", None) == "pi05":
        return getattr(policy_cfg, "ee_state_key", "observation.ee_state") or "observation.ee_state"
    return "observation.state"


def _to_hwc_if_image(key: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if "image" in key and arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return np.transpose(arr, (1, 2, 0))
    return arr


def _to_policy_observation(obs: dict[str, Any], rename_map: dict[str, str]) -> dict[str, np.ndarray]:
    client_obs = _openpi_obs_to_lerobot(obs)
    policy_obs: dict[str, np.ndarray] = {}
    for client_key, value in client_obs.items():
        policy_key = rename_map.get(client_key, client_key)
        policy_obs[policy_key] = _to_hwc_if_image(policy_key, np.array(_as_numpy(value), copy=True))
    return policy_obs


class LongLivedWebsocketPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """OpenPI server with relaxed keepalive for long robot trajectory gaps."""

    def __init__(
        self,
        *,
        ping_interval: int = DEFAULT_PING_INTERVAL_S,
        ping_timeout: int = DEFAULT_PING_TIMEOUT_S,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

    async def run(self) -> None:
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=websocket_policy_server._health_check,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        ) as server:
            await server.serve_forever()


class LeRobotOpenPIBridge(_base_policy.BasePolicy):
    def __init__(self, session: _PolicySession, rename_map: dict[str, str]):
        self._session = session
        self._rename_map = rename_map
        self._policy_cfg = session.policy_cfg
        self._logged_observation_schema = False

    def _validate_and_log_observation(self, observation: dict[str, np.ndarray]) -> None:
        required = _required_policy_observation_keys(self._policy_cfg)
        missing = sorted(required.difference(observation))
        if missing:
            raise ValueError(f"Missing required policy observations after rename: {missing}")

        state_key = _response_state_key(self._policy_cfg)
        state = observation[state_key]
        required_state_dim = 14
        if state.ndim != 1 or state.shape[-1] < required_state_dim:
            raise ValueError(
                f"Expected {state_key} shape [D>={required_state_dim}] for "
                f"dual-arm xyz+quaternion EE pose, got {state.shape}"
            )

        if getattr(self._policy_cfg, "type", None) == "pi0_dmp":
            ref_state = observation["observation.reference.state"]
            ref_actions = observation["observation.ref_actions"]
            if ref_state.ndim != 1 or ref_state.shape[-1] < required_state_dim:
                raise ValueError(
                    f"Expected observation.reference.state shape [D>={required_state_dim}], "
                    f"got {ref_state.shape}"
                )
            if ref_actions.ndim != 2 or ref_actions.shape[-1] < required_state_dim:
                raise ValueError(
                    f"Expected observation.ref_actions shape [T,D>={required_state_dim}], "
                    f"got {ref_actions.shape}"
                )

        if not self._logged_observation_schema:
            schema = {
                key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(observation.items())
            }
            logging.info(
                "First OpenPI observation after rename (policy.type=%s): %s",
                getattr(self._policy_cfg, "type", None),
                schema,
            )
            logging.info(
                "Model-facing EE layout: dual-arm xyz+Rot6D pose plus raw tail, clip/pad32 "
                "(raw quaternion EE state retained for decoding)"
            )
            self._logged_observation_schema = True

    @override
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        payload = _unwrap_client_payload(obs)
        task = payload.get("prompt", payload.get("task"))
        if task is not None and not isinstance(task, str):
            task = str(task)

        observation = _to_policy_observation(payload, self._rename_map)
        self._validate_and_log_observation(observation)
        actions = self._session.infer_action_chunk(task=task, observation=observation)
        actions_np = _as_numpy(actions)

        response: dict[str, Any] = {"actions": actions_np}
        state_key = _response_state_key(self._policy_cfg)
        if state_key in observation:
            response["state"] = observation[state_key]
        elif "observation.state" in observation:
            response["state"] = observation["observation.state"]
        elif "state" in payload:
            response["state"] = _as_numpy(payload["state"])
        return response


@dataclass
class OpenPIBridgeConfig:
    policy: PreTrainedConfig | None = None
    websocket: WebsocketServerConfig = field(default_factory=WebsocketServerConfig)
    robot_type: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    port: int = 8008
    ping_interval: int = DEFAULT_PING_INTERVAL_S
    ping_timeout: int = DEFAULT_PING_TIMEOUT_S
    pi0_dmp_kv_cache: bool = False

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("You must pass --policy.path=... (checkpoint dir or Hub model id).")
        cli_overrides = parser.get_cli_overrides("policy")
        policy_path = _resolve_pretrained_model_path(policy_path)
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path


@parser.wrap()
def serve_openpi_bridge(cfg: OpenPIBridgeConfig) -> None:
    init_logging()
    assert cfg.policy is not None

    infer_cfg = PolicyInferWebsocketConfig(
        policy=cfg.policy,
        websocket=cfg.websocket,
        robot_type=cfg.robot_type,
        rename_map=cfg.rename_map,
    )
    session = _PolicySession(infer_cfg)
    if cfg.pi0_dmp_kv_cache:
        if cfg.policy.type != "pi0_dmp":
            raise ValueError("--pi0_dmp_kv_cache=true is only valid for a pi0_dmp checkpoint.")
        _enable_pi0_dmp_kv_cache(session.policy)
    bridge = LeRobotOpenPIBridge(session, rename_map=cfg.rename_map)

    port = cfg.port if cfg.port is not None else cfg.websocket.port
    logging.info(
        "LeRobot checkpoint %s served as OpenPI msgpack WebSocket on ws://0.0.0.0:%s "
        "(ping_interval=%ss, ping_timeout=%ss)",
        cfg.policy.pretrained_path,
        port,
        cfg.ping_interval,
        cfg.ping_timeout,
    )

    server = LongLivedWebsocketPolicyServer(
        policy=bridge,
        host="0.0.0.0",
        port=port,
        ping_interval=cfg.ping_interval,
        ping_timeout=cfg.ping_timeout,
        metadata={
            "policy_type": cfg.policy.type,
            "protocol": "openpi_msgpack",
            "checkpoint": str(cfg.policy.pretrained_path),
        },
    )
    server.serve_forever()


def main() -> None:
    register_third_party_plugins()
    serve_openpi_bridge()


if __name__ == "__main__":
    main()
