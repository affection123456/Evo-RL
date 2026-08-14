#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run **policy inference only** behind a small WebSocket JSON API (no robot / teleop).

Install WebSocket dependency::

    pip install "lerobot[policy-infer]"

Example::
    export MODEL_ZOO=/mnt/data/modelzoo
    lerobot-policy-infer \
      --policy.path=/mnt/nas/wanghao/openpi_05/Evo-RL/outputs/train/0512_1/checkpoints/last \
      --policy.device=cuda \
      --websocket.host=0.0.0.0 \
      --websocket.port=8004

Client message (JSON text)::

    {"task": "...", "observation": {"observation.state": [...], ...}}

Send ``{"type": "spec"}`` for required observation keys.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext
from copy import copy, deepcopy
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import get_safe_torch_device, init_logging


def _resolve_pretrained_model_path(policy_path: str) -> str:
    """If ``--policy.path`` points at a train step dir, use ``.../pretrained_model``."""
    raw = policy_path.strip()
    p = Path(raw).expanduser()
    # Local-looking paths must exist; do not fall through to Hub (HFValidationError).
    looks_local = (
        raw.startswith((".", "/", "~"))
        or raw.startswith("outputs/")
        or "/" in raw
        or "\\" in raw
        or p.suffix in {".json", ".safetensors"}
    )
    if looks_local and not p.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {p.resolve() if p.is_absolute() else p}. "
            "Training may not have saved checkpoints yet, or the run dir was overwritten. "
            "Pass an existing .../checkpoints/<step> or .../checkpoints/last "
            f"(with {PRETRAINED_MODEL_DIR}/config.json), or a Hub repo id like 'org/name'."
        )
    if not p.exists():
        return raw
    p = p.resolve()
    if (p / "config.json").is_file():
        return str(p)
    nested = p / PRETRAINED_MODEL_DIR
    if (nested / "config.json").is_file():
        logging.info("Resolved --policy.path to %s", nested)
        return str(nested)
    raise FileNotFoundError(
        f"No config.json under {p} or {nested}. "
        f"For training checkpoints use .../checkpoints/<step>/{PRETRAINED_MODEL_DIR} "
        "or pass the step directory; this script appends pretrained_model when missing."
    )


def _require_websockets():
    try:
        return __import__("websockets")
    except ImportError as e:
        raise ImportError(
            "The `websockets` package is required. Install with:\n"
            '  pip install "lerobot[policy-infer]"\n'
            "or: pip install websockets"
        ) from e


def _invert_rename(rename_map: dict[str, str]) -> dict[str, str]:
    return {new: old for old, new in rename_map.items()}


def _decode_image_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        raw = base64.b64decode(value)
        with Image.open(BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    if isinstance(value, list):
        arr = np.asarray(value, dtype=np.uint8)
        if arr.ndim != 3:
            raise ValueError(f"Image list must decode to HWC uint8, got shape {arr.shape}")
        return arr
    raise TypeError(f"Unsupported image payload type: {type(value)}")


def _decode_numeric_or_token_list(val: Any, *, as_language: bool) -> np.ndarray:
    if not isinstance(val, list):
        raise TypeError(f"Expected JSON list, got {type(val)}")
    if as_language and val:

        def _looks_like_token_ids() -> bool:
            for x in val:
                if isinstance(x, bool):
                    return False
                if isinstance(x, int):
                    continue
                if isinstance(x, float) and x.is_integer():
                    continue
                return False
            return True

        if _looks_like_token_ids():
            return np.asarray(val, dtype=np.int64)
    return np.asarray(val, dtype=np.float32)


def _decode_observation_payload(
    raw: dict[str, Any],
    policy_cfg: PreTrainedConfig,
    rename_map: dict[str, str],
) -> dict[str, np.ndarray]:
    obs_in = raw.get("observation")
    if not isinstance(obs_in, dict):
        raise ValueError("Missing or invalid 'observation' object.")

    inv = _invert_rename(rename_map)
    out: dict[str, np.ndarray] = {}
    for policy_key, ft in (policy_cfg.input_features or {}).items():
        if ft.type not in (FeatureType.STATE, FeatureType.VISUAL, FeatureType.LANGUAGE):
            continue
        client_key = inv.get(policy_key, policy_key)
        if client_key not in obs_in:
            raise KeyError(
                f"Missing observation for policy feature '{policy_key}': "
                f"expected JSON key '{client_key}'."
            )
        val = obs_in[client_key]
        is_visual = ft.type == FeatureType.VISUAL or "image" in policy_key or "image" in client_key
        if is_visual:
            decoded = np.ascontiguousarray(_decode_image_value(val))
        elif ft.type == FeatureType.LANGUAGE:
            decoded = _decode_numeric_or_token_list(val, as_language=True)
        elif isinstance(val, list):
            decoded = np.asarray(val, dtype=np.float32).copy()
        else:
            raise TypeError(f"Key '{client_key}': expected JSON list, got {type(val)}")
        out[client_key] = decoded
        if client_key != policy_key:
            # Some policy-specific preprocessors (e.g. PI0-DMP) pack canonical keys directly
            # instead of using a generic rename processor.
            out[policy_key] = decoded
    return out


def _spec_payload(policy_cfg: PreTrainedConfig, rename_map: dict[str, str]) -> dict[str, Any]:
    inv = _invert_rename(rename_map)
    hints: dict[str, str] = {}
    required_client_keys: list[str] = []
    for policy_key, ft in (policy_cfg.input_features or {}).items():
        if ft.type not in (FeatureType.STATE, FeatureType.VISUAL, FeatureType.LANGUAGE):
            continue
        client_key = inv.get(policy_key, policy_key)
        required_client_keys.append(client_key)
        if ft.type == FeatureType.VISUAL or "image" in policy_key or "image" in client_key:
            hints[client_key] = "base64_png_or_hwc_uint8_list"
        elif ft.type == FeatureType.STATE:
            hints[client_key] = "float32_vector_json_list"
        elif ft.type == FeatureType.LANGUAGE:
            hints[client_key] = "token_ids_int_list_or_float_tensor_list"
    return {
        "type": "spec",
        "policy_type": policy_cfg.type,
        "rename_map": rename_map,
        "required_client_keys": required_client_keys,
        "observation_keys": hints,
    }


def _clone_cache(cache: Any) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache") and hasattr(cache.__class__, "from_legacy_cache"):
        return cache.__class__.from_legacy_cache(cache.to_legacy_cache())
    return deepcopy(cache)


def _patch_pi05_inference_compat(policy: PreTrainedPolicy) -> None:
    """Keep PI05 sampling compatible with the current Transformers Gemma cache/dtype behavior."""
    if getattr(policy.config, "type", None) != "pi05" or not hasattr(policy, "model"):
        return

    model = policy.model
    if getattr(model, "_policy_infer_pi05_patched", False):
        return

    original_embed_prefix = model.embed_prefix
    original_embed_suffix = model.embed_suffix
    original_denoise_step = model.denoise_step

    def _prefix_dtype() -> torch.dtype:
        return model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype

    def _suffix_dtype() -> torch.dtype:
        return model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype

    def embed_prefix(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs, pad_masks, att_masks = original_embed_prefix(*args, **kwargs)
        return embs.to(dtype=_prefix_dtype()), pad_masks, att_masks

    def embed_suffix(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embs, pad_masks, att_masks, adarms_cond = original_embed_suffix(*args, **kwargs)
        return embs.to(dtype=_suffix_dtype()), pad_masks, att_masks, adarms_cond

    def denoise_step(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Gemma DynamicCache is mutated by attention even when use_cache=False; PI05 expects
        # each flow-matching step to reuse the same prefix cache, not append suffix tokens to it.
        if "past_key_values" in kwargs:
            kwargs = dict(kwargs)
            kwargs["past_key_values"] = _clone_cache(kwargs["past_key_values"])
            return original_denoise_step(*args, **kwargs)
        if len(args) >= 2:
            args = list(args)
            args[1] = _clone_cache(args[1])
            return original_denoise_step(*args, **kwargs)
        return original_denoise_step(*args, **kwargs)

    model.embed_prefix = embed_prefix
    model.embed_suffix = embed_suffix
    model.denoise_step = denoise_step
    model._policy_infer_pi05_patched = True


@dataclass
class WebsocketServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    # Default websockets limit is 1 MiB; multi-camera PNG JSON easily exceeds that.
    max_size_mb: int = 32


@dataclass
class PolicyInferWebsocketConfig:
    policy: PreTrainedConfig | None = None
    websocket: WebsocketServerConfig = field(default_factory=WebsocketServerConfig)
    robot_type: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)

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


class _PolicySession:
    def __init__(self, cfg: PolicyInferWebsocketConfig):
        assert cfg.policy is not None
        self.policy_cfg = cfg.policy
        self.robot_type = cfg.robot_type
        self.rename_map = cfg.rename_map or {}
        self.device = get_safe_torch_device(cfg.policy.device, log=True)
        policy_cls = get_policy_class(cfg.policy.type)
        self.policy: PreTrainedPolicy = policy_cls.from_pretrained(
            cfg.policy.pretrained_path,
            config=cfg.policy,
        )
        self.policy.eval()
        _patch_pi05_inference_compat(self.policy)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": self.rename_map},
            },
        )

    def infer(self, task: str | None, observation: dict[str, np.ndarray]) -> PolicyAction:
        # PI05 checkpoints use dtype=bfloat16 while use_amp=false; training still runs under accelerate bf16.
        # predict_action's autocast() defaults to fp16 on CUDA — use explicit bf16 autocast instead.
        dtype = getattr(self.policy_cfg, "dtype", None)
        if self.device.type == "cuda" and dtype in ("bfloat16", "bf16"):
            amp_ctx: Any = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            use_amp = False
        elif self.device.type == "cuda" and self.policy_cfg.use_amp:
            amp_ctx = torch.autocast(device_type="cuda")
            use_amp = False
        else:
            amp_ctx = nullcontext()
            use_amp = bool(self.policy_cfg.use_amp)

        with amp_ctx:
            action = predict_action(
                observation=copy(observation),
                policy=self.policy,
                device=self.device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=use_amp,
                task=task,
                robot_type=self.robot_type,
            )
            return _decode_policy_actions(action, observation, self.policy_cfg)

    def infer_action_chunk(self, task: str | None, observation: dict[str, np.ndarray]) -> PolicyAction:
        dtype = getattr(self.policy_cfg, "dtype", None)
        if self.device.type == "cuda" and dtype in ("bfloat16", "bf16"):
            amp_ctx: Any = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            use_amp = False
        elif self.device.type == "cuda" and self.policy_cfg.use_amp:
            amp_ctx = torch.autocast(device_type="cuda")
            use_amp = False
        else:
            amp_ctx = nullcontext()
            use_amp = bool(self.policy_cfg.use_amp)

        with (
            torch.inference_mode(),
            amp_ctx,
            torch.autocast(device_type=self.device.type) if self.device.type == "cuda" and use_amp else nullcontext(),
        ):
            from lerobot.policies.utils import prepare_observation_for_inference

            processed = prepare_observation_for_inference(copy(observation), self.device, task, self.robot_type)
            processed = self.preprocessor(processed)
            action = self.policy.predict_action_chunk(processed)
            action = self.postprocessor(action)
            action = _decode_policy_actions(action, observation, self.policy_cfg)
            if isinstance(action, torch.Tensor) and action.ndim >= 3 and action.shape[0] == 1:
                action = action[0]
            return action


def _action_to_jsonable(action: PolicyAction) -> Any:
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().tolist()
    return action


def _decode_policy_actions(
    actions: PolicyAction,
    raw_observation: dict[str, np.ndarray],
    policy_cfg: PreTrainedConfig,
) -> PolicyAction:
    if not isinstance(actions, torch.Tensor):
        return actions

    policy_type = getattr(policy_cfg, "type", None)
    if policy_type == "pi0_dmp":
        from lerobot.policies.pi0_dmp.processor_pi0_dmp import decode_pi0_dmp_policy_actions

        return decode_pi0_dmp_policy_actions(actions, raw_observation, policy_cfg)
    if policy_type == "pi05":
        from lerobot.policies.pi05.processor_pi05 import decode_pi05_policy_actions

        return decode_pi05_policy_actions(actions, raw_observation, policy_cfg)
    return actions


@parser.wrap()
def policy_infer_websocket(cfg: PolicyInferWebsocketConfig):
    init_logging()
    websockets = _require_websockets()

    session = _PolicySession(cfg)
    host = cfg.websocket.host
    port = cfg.websocket.port
    logging.info("Policy loaded from %s; WebSocket on ws://%s:%s", cfg.policy.pretrained_path, host, port)

    async def handler(websocket: Any) -> None:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({"ok": False, "error": f"invalid_json: {e}"}))
                continue

            if data.get("type") == "spec":
                await websocket.send(
                    json.dumps({"ok": True, **_spec_payload(session.policy_cfg, session.rename_map)})
                )
                continue

            try:
                task = data.get("task")
                obs = _decode_observation_payload(data, session.policy_cfg, session.rename_map)
                if data.get("return_chunk", False):
                    action = session.infer_action_chunk(
                        task=str(task) if task is not None else None, observation=obs
                    )
                    response_key = "actions"
                else:
                    action = session.infer(task=str(task) if task is not None else None, observation=obs)
                    response_key = "action"
                logging.info("Infer action: %s", _action_to_jsonable(action))
                await websocket.send(
                    json.dumps({"ok": True, response_key: _action_to_jsonable(action)}, allow_nan=False)
                )
            except Exception as e:
                logging.exception("Inference error")
                await websocket.send(json.dumps({"ok": False, "error": str(e)}))

    async def runner() -> None:
        max_size = cfg.websocket.max_size_mb * 1024 * 1024
        async with websockets.serve(handler, host, port, max_size=max_size):
            await asyncio.Future()

    asyncio.run(runner())


def main() -> None:
    register_third_party_plugins()
    policy_infer_websocket()


if __name__ == "__main__":
    main()
