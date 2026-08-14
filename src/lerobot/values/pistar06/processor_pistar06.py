#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import create_transition, policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    REWARD,
    TRUNCATED,
)
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config

PISTAR06_IMAGES_KEY = "observation.pistar06.images"
PISTAR06_IMAGE_MASK_KEY = "observation.pistar06.image_attention_mask"


# Bare DMP video/state keys → canonical LeRobot observation.* names.
_PISTAR06_RAW_TO_CANONICAL = {
    "ee_state": OBS_STATE,
    "ref_ee_state": "observation.reference.state",
    "top_head": f"{OBS_IMAGES}.top_head",
    "hand_right": f"{OBS_IMAGES}.hand_right",
    "hand_left": f"{OBS_IMAGES}.hand_left",
    "ref_top_head": f"{OBS_IMAGES}.ref_top_head",
    "ref_hand_right": f"{OBS_IMAGES}.ref_hand_right",
    "ref_hand_left": f"{OBS_IMAGES}.ref_hand_left",
}


def pistar06_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    control_keys = {ACTION, REWARD, DONE, TRUNCATED, "info"}
    complementary_keys = {"task", "subtask", "index", "task_index", "episode_index", "frame_index"}
    observation = {key: value for key, value in batch.items() if key not in control_keys | complementary_keys}
    for raw_key, canonical_key in _PISTAR06_RAW_TO_CANONICAL.items():
        if raw_key in batch and canonical_key not in observation:
            observation[canonical_key] = batch[raw_key]
    complementary_data = {key: batch[key] for key in complementary_keys if key in batch}
    return create_transition(
        observation=observation,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data,
    )


def _quat_to_matrix(quat: Tensor) -> Tensor:
    quat = quat.to(dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _matrix_first_two_cols_to_rot6d(matrix: Tensor) -> Tensor:
    """Flatten the first two rotation-matrix columns as ``[col0, col1]``."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _quat_pose_to_rot6d(x: Tensor) -> Tensor:
    if x.shape[-1] < 14:
        raise ValueError(f"Expected last dim >= 14 for dual-arm xyz+quat layout, got {tuple(x.shape)}")
    left = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 3:7]))
    right = _matrix_first_two_cols_to_rot6d(_quat_to_matrix(x[..., 10:14]))
    return torch.cat([x[..., :3], left, x[..., 7:10], right, x[..., 14:]], dim=-1)


def _pad_or_clip_last_dim(x: Tensor, dim: int) -> Tensor:
    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        return functional.pad(x, (0, dim - x.shape[-1]))
    return x


def _pad_last_dim(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return functional.pad(vector, (0, new_dim - vector.shape[-1]))


@ProcessorStepRegistry.register(name="pistar06_rot6d_state_processor")
@dataclass
class Pistar06Rot6DStateProcessorStep(ProcessorStep):
    state_feature: str = OBS_STATE
    ref_state_feature: str = "observation.reference.state"
    max_state_dim: int = 32
    use_rot6d: bool = True

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature": self.state_feature,
            "ref_state_feature": self.ref_state_feature,
            "max_state_dim": self.max_state_dim,
            "use_rot6d": self.use_rot6d,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        for key in (self.state_feature, self.ref_state_feature):
            if key in observation:
                value = torch.as_tensor(observation[key])
                if self.use_rot6d:
                    value = _quat_pose_to_rot6d(value)
                observation[key] = _pad_or_clip_last_dim(value, self.max_state_dim)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pistar06_prepare_task_prompt")
@dataclass
class Pistar06PrepareTaskPromptProcessorStep(ProcessorStep):
    task_key: str = "task"
    include_state_in_prompt: bool = True
    include_ref_state_in_prompt: bool = True
    state_feature: str = OBS_STATE
    ref_state_feature: str = "observation.reference.state"
    max_state_dim: int = 32
    state_discretization_bins: int = 256

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "include_state_in_prompt": self.include_state_in_prompt,
            "include_ref_state_in_prompt": self.include_ref_state_in_prompt,
            "state_feature": self.state_feature,
            "ref_state_feature": self.ref_state_feature,
            "max_state_dim": self.max_state_dim,
            "state_discretization_bins": self.state_discretization_bins,
        }

    @staticmethod
    def _clean_prompt(task: str) -> str:
        return str(task).strip().replace("_", " ").replace("\n", " ").strip()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.task_key not in complementary_data:
            raise KeyError(f"Missing task field '{self.task_key}' in complementary data.")
        tasks_raw = complementary_data[self.task_key]
        if isinstance(tasks_raw, str):
            tasks = [tasks_raw]
        elif isinstance(tasks_raw, Sequence) and all(isinstance(task, str) for task in tasks_raw):
            tasks = list(tasks_raw)
        else:
            raise TypeError(
                f"Expected task field '{self.task_key}' as sequence of strings, got {type(tasks_raw)}."
            )

        def discretize_state(feature: str, *, required: bool) -> np.ndarray | None:
            if feature not in observation:
                if required:
                    raise KeyError(f"Missing state feature '{feature}' while building value prompt.")
                return None
            state = observation[feature]
            if not isinstance(state, Tensor):
                state = torch.as_tensor(state)

            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.ndim != 2:
                raise ValueError(
                    f"Expected state tensor with shape [B, D], got {tuple(state.shape)} "
                    f"for feature '{feature}'."
                )

            state = state.detach().to(dtype=torch.float32, device="cpu")
            state = _pad_last_dim(state, self.max_state_dim)
            state_np = state.numpy()
            bins = np.linspace(-1.0, 1.0, self.state_discretization_bins + 1, dtype=np.float32)[:-1]
            return np.digitize(state_np, bins=bins) - 1

        prompts: list[str] = []
        if self.include_state_in_prompt:
            discretized_state = discretize_state(self.state_feature, required=True)
            discretized_ref_state = (
                discretize_state(self.ref_state_feature, required=False)
                if self.include_ref_state_in_prompt
                else None
            )
            if discretized_state is None:
                raise RuntimeError("State discretization unexpectedly returned None.")

            if discretized_state.shape[0] != len(tasks):
                raise ValueError(
                    f"Task count ({len(tasks)}) does not match state batch size ({discretized_state.shape[0]})."
                )
            if discretized_ref_state is not None and discretized_ref_state.shape[0] != len(tasks):
                raise ValueError(
                    f"Task count ({len(tasks)}) does not match ref state batch size ({discretized_ref_state.shape[0]})."
                )

            for i, task in enumerate(tasks):
                cleaned_task = self._clean_prompt(task)
                state_str = " ".join(map(str, discretized_state[i].tolist()))
                if discretized_ref_state is None:
                    prompts.append(f"Task: {cleaned_task}, State: {state_str}\nValue: ")
                else:
                    ref_state_str = " ".join(map(str, discretized_ref_state[i].tolist()))
                    prompts.append(
                        f"Task: {cleaned_task}, State: {state_str}, Reference State: {ref_state_str}\nValue: "
                    )
        else:
            prompts = [f"Task: {self._clean_prompt(task)}\nValue: " for task in tasks]

        complementary_data[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pistar06_prepare_images")
@dataclass
class Pistar06PrepareImagesProcessorStep(ProcessorStep):
    camera_features: list[str]

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_features": self.camera_features,
        }

    @staticmethod
    def _to_bchw(img_batch: Tensor) -> Tensor:
        if img_batch.ndim != 4:
            raise ValueError(f"Expected image batch rank 4, got shape {tuple(img_batch.shape)}.")

        if img_batch.shape[1] in {1, 3}:  # [B,C,H,W]
            return img_batch
        if img_batch.shape[-1] in {1, 3}:  # [B,H,W,C]
            return img_batch.permute(0, 3, 1, 2)
        raise ValueError(
            "Camera tensor must be channels-first or channels-last. "
            f"Got camera batch with shape={tuple(img_batch.shape)}."
        )

    def _process_camera_batch(self, img_batch: Tensor) -> Tensor:
        return self._to_bchw(img_batch).detach().to(dtype=torch.float32)

    @staticmethod
    def _match_spatial_size(img: Tensor, ref_hw: tuple[int, int]) -> Tensor:
        """Align [B,C,H,W] to ref (H,W) by center crop / center zero-pad (no resize).

        Example: H 960 -> 720 crops 120 rows from the top and 120 from the bottom.
        """
        rh, rw = ref_hw
        if img.shape[2] == rh and img.shape[3] == rw:
            return img
        b, c, h, w = img.shape
        img = img.float()
        out = torch.zeros(b, c, rh, rw, device=img.device, dtype=img.dtype)

        if h >= rh:
            t0 = (h - rh) // 2
            t1 = t0 + rh
            dt0, dt1 = 0, rh
        else:
            t0, t1 = 0, h
            dt0 = (rh - h) // 2
            dt1 = dt0 + h

        if w >= rw:
            l0 = (w - rw) // 2
            l1 = l0 + rw
            dl0, dl1 = 0, rw
        else:
            l0, l1 = 0, w
            dl0 = (rw - w) // 2
            dl1 = dl0 + w

        out[:, :, dt0:dt1, dl0:dl1] = img[:, :, t0:t1, l0:l1]
        return out

    def _prepare_images(self, observation: dict[str, Any]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.camera_features if key in observation]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(observation.keys())}"
            )

        reference_img = self._process_camera_batch(torch.as_tensor(observation[present_img_keys[0]]))
        ref_hw = (int(reference_img.shape[2]), int(reference_img.shape[3]))
        bsize = reference_img.shape[0]
        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in observation:
                img = self._process_camera_batch(torch.as_tensor(observation[key]))
                if img.shape[0] != bsize:
                    raise ValueError(
                        f"Mismatched batch size across cameras. Camera '{key}' has {img.shape[0]}, expected {bsize}."
                    )
                if img.shape[1] != reference_img.shape[1]:
                    raise ValueError(
                        "Camera tensors must share the same channel count C before model preprocessing. "
                        f"Camera '{key}' has C={img.shape[1]}, expected C={reference_img.shape[1]}."
                    )
                img = self._match_spatial_size(img, ref_hw)
                image_tensors.append(img)
                image_masks.append(torch.ones(bsize, dtype=torch.bool))
            else:
                image_tensors.append(torch.zeros_like(reference_img))
                image_masks.append(torch.zeros(bsize, dtype=torch.bool))

        images = torch.stack(image_tensors, dim=1)
        masks = torch.stack(image_masks, dim=1)
        return images, masks

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        images, image_attention_mask = self._prepare_images(observation)
        observation[PISTAR06_IMAGES_KEY] = images.to(dtype=torch.float32)
        observation[PISTAR06_IMAGE_MASK_KEY] = image_attention_mask.to(dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pistar06_pre_post_processors(
    config: Pistar06Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,  # noqa: ARG001
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        for key in config.input_features or {}:
            if key.startswith(f"{OBS_IMAGES}."):
                camera_features.append(key)
            elif key in _PISTAR06_RAW_TO_CANONICAL and not key.startswith("ref_"):
                # DMP meta exposes bare video keys; map to observation.images.*.
                # Reference cameras are appended from config.reference_camera_features.
                canon = _PISTAR06_RAW_TO_CANONICAL[key]
                if canon.startswith(f"{OBS_IMAGES}.") and canon not in camera_features:
                    camera_features.append(canon)
    for key in config.reference_camera_features:
        if key not in camera_features:
            camera_features.append(key)

    processor_features = {**(config.input_features or {}), **(config.output_features or {})}
    for key in (config.state_feature, config.ref_state_feature):
        if key and key not in processor_features:
            stat_shape = None
            if dataset_stats and key in dataset_stats and "mean" in dataset_stats[key]:
                stat_shape = tuple(torch.as_tensor(dataset_stats[key]["mean"]).shape)
            processor_features[key] = PolicyFeature(
                type=FeatureType.STATE,
                shape=stat_shape if stat_shape else (config.max_state_dim,),
            )

    normalize_observation_keys = {config.state_feature}
    if config.include_ref_state_in_prompt and config.ref_state_feature:
        normalize_observation_keys.add(config.ref_state_feature)

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        Pistar06Rot6DStateProcessorStep(
            state_feature=config.state_feature,
            ref_state_feature=config.ref_state_feature,
            max_state_dim=config.max_state_dim,
            use_rot6d=bool(getattr(config, "use_rot6d", True)),
        ),
        NormalizerProcessorStep(
            features=processor_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys=normalize_observation_keys,
        ),
        Pistar06PrepareTaskPromptProcessorStep(
            task_key=config.task_field,
            include_state_in_prompt=config.include_state_in_prompt,
            include_ref_state_in_prompt=config.include_ref_state_in_prompt,
            state_feature=config.state_feature,
            ref_state_feature=config.ref_state_feature,
            max_state_dim=config.max_state_dim,
            state_discretization_bins=config.state_discretization_bins,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.language_repo_id,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        Pistar06PrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=pistar06_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
