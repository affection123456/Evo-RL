#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.ee_action_contract import (
    EEActionContract,
    make_ee_action_contract,
    make_raw_ee_action_template,
    pack_ee_tensor,
    pad_or_clip_ee,
    unpack_ee_tensor,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    _extract_complementary_data,
    create_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

OBS_TOP_HEAD = f"{OBS_IMAGES}.top_head"
OBS_HAND_RIGHT = f"{OBS_IMAGES}.hand_right"
OBS_HAND_LEFT = f"{OBS_IMAGES}.hand_left"
OBS_EE_STATE = "observation.ee_state"
OBS_EE_ACTIONS = "observation.ee_actions"


def _first_present(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def pi05_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Pack basket ``observation.*`` and DMP bare keys into a pi05 observation dict.

    Standard ``batch_to_transition`` only keeps keys with the ``observation.`` prefix, so
    DMP fields like ``ee_state`` / ``top_head`` never become an observation and the
    preprocessor fails. This converter accepts both layouts.
    """
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    observation = {key: value for key, value in batch.items() if key.startswith(OBS_PREFIX)}

    for canonical_key, aliases in (
        (OBS_EE_STATE, (OBS_EE_STATE, "ee_state")),
        (OBS_EE_ACTIONS, (OBS_EE_ACTIONS, "ee_actions")),
        (OBS_TOP_HEAD, (OBS_TOP_HEAD, "top_head")),
        (OBS_HAND_RIGHT, (OBS_HAND_RIGHT, "hand_right")),
        (OBS_HAND_LEFT, (OBS_HAND_LEFT, "hand_left")),
    ):
        if canonical_key not in observation:
            value = _first_present(batch, *aliases)
            if value is not None:
                observation[canonical_key] = value

    complementary_data = _extract_complementary_data(batch)
    return create_transition(
        observation=observation if observation else None,
        action=batch.get(ACTION),
        complementary_data=complementary_data if complementary_data else None,
    )


@ProcessorStepRegistry.register(name="pi05_rot6d_delta_processor")
@dataclass
class Pi05Rot6DDeltaProcessorStep(ProcessorStep):
    """Map raw EE pose → selected canonical state/action contract.

    Selection and raw indices come from ``ee_action_contract``. ``use_rot6d``
    chooses xyz+quat+gripper (8D right1) or xyz+rot6d+gripper (10D right1).
    The selected physical vector is then padded to the model head width.
    """

    state_dim: int = 32
    action_dim: int = 32
    ee_state_key: str = "observation.ee_state"
    ee_action_key: str = "observation.ee_actions"
    use_delta: bool = False
    delta_dims: int | None = None
    use_rot6d: bool = True
    arm_mode: str = "right"
    gripper_dims: int = 1

    def get_config(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "ee_state_key": self.ee_state_key,
            "ee_action_key": self.ee_action_key,
            "use_delta": self.use_delta,
            "delta_dims": self.delta_dims,
            "use_rot6d": self.use_rot6d,
            "arm_mode": self.arm_mode,
            "gripper_dims": self.gripper_dims,
        }

    def _contract(self) -> EEActionContract:
        return make_ee_action_contract(
            use_rot6d=self.use_rot6d,
            arm_mode=self.arm_mode,
            gripper_dims=self.gripper_dims,
        )

    def _pose_delta_dims(self) -> int:
        if self.delta_dims is not None:
            return self.delta_dims
        return self._contract().pose_dim

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        raw_state = _first_present(observation, self.ee_state_key, "ee_state", OBS_STATE)
        if raw_state is None:
            raise KeyError(
                f"pi05 Rot6D requires EE quat state; missing any of "
                f"{self.ee_state_key!r}, 'ee_state', {OBS_STATE!r}"
            )
        contract = self._contract()
        state = pad_or_clip_ee(pack_ee_tensor(torch.as_tensor(raw_state), contract), self.state_dim)

        # Prefer EE quat actions; dataset `action` is often joint-space and must not be used.
        raw_action = _first_present(observation, self.ee_action_key, "ee_actions")
        if raw_action is None:
            raw_action = transition.get(TransitionKey.ACTION)
        actions = (
            None
            if raw_action is None
            else pad_or_clip_ee(
                pack_ee_tensor(torch.as_tensor(raw_action), contract),
                self.action_dim,
            )
        )

        if self.use_delta and actions is not None:
            delta_dims = self._pose_delta_dims()
            dims = min(delta_dims, self.action_dim, self.state_dim)
            if dims > 0:
                actions = actions.clone()
                state_for_delta = state
                if actions.ndim == state.ndim + 1:
                    state_for_delta = state.unsqueeze(-2)
                actions[..., :dims] = actions[..., :dims] - state_for_delta[..., :dims]

        observation[OBS_STATE] = state
        transition[TransitionKey.OBSERVATION] = observation
        if actions is not None:
            transition[TransitionKey.ACTION] = actions
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def decode_pi05_policy_actions(
    actions: torch.Tensor,
    raw_observation: dict[str, Any],
    policy_cfg: PI05Config | Any,
) -> torch.Tensor:
    """Decode selected model output and scatter it into a full raw EE action."""
    contract = make_ee_action_contract(
        use_rot6d=bool(getattr(policy_cfg, "use_rot6d", False)),
        arm_mode=getattr(policy_cfg, "ee_arm_mode", "right"),
        gripper_dims=int(getattr(policy_cfg, "ee_gripper_dims", 1)),
    )
    decoded = actions
    state_value = _first_present(
        raw_observation,
        getattr(policy_cfg, "ee_state_key", "observation.ee_state"),
        "observation.ee_state",
        "ee_state",
        OBS_STATE,
        "observation/state",
    )
    if getattr(policy_cfg, "rot6d_delta_action", False) and state_value is not None:
        max_state_dim = getattr(policy_cfg, "max_state_dim", actions.shape[-1])
        max_action_dim = getattr(policy_cfg, "max_action_dim", actions.shape[-1])
        delta_dims = min(contract.pose_dim, max_action_dim, max_state_dim)

        state = pad_or_clip_ee(
            pack_ee_tensor(torch.as_tensor(state_value), contract),
            max_state_dim,
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(device=actions.device, dtype=actions.dtype)

        decoded = actions.clone()
        if delta_dims > 0:
            state_for_delta = state
            if decoded.ndim == state.ndim + 1:
                state_for_delta = state.unsqueeze(-2)
            decoded[..., :delta_dims] = decoded[..., :delta_dims] + state_for_delta[..., :delta_dims]

    ee_action = _first_present(
        raw_observation,
        getattr(policy_cfg, "ee_action_key", "observation.ee_actions"),
        "observation.ee_actions",
        "ee_actions",
        ACTION,
    )
    if state_value is None:
        raise KeyError("EE contract decode needs raw EE state as a full-action template.")
    if ee_action is not None:
        action_dim = int(torch.as_tensor(ee_action).shape[-1])
    else:
        action_dim = int(
            getattr(policy_cfg, "ee_raw_action_dim", torch.as_tensor(state_value).shape[-1])
        )
    template = make_raw_ee_action_template(torch.as_tensor(state_value), action_dim)
    return unpack_ee_tensor(decoded, template, contract)


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    contract_steps: list[ProcessorStep] = [
        Pi05Rot6DDeltaProcessorStep(
            state_dim=config.max_state_dim,
            action_dim=config.max_action_dim,
            ee_state_key=config.ee_state_key,
            ee_action_key=config.ee_action_key,
            use_delta=bool(getattr(config, "rot6d_delta_action", False)),
            use_rot6d=bool(getattr(config, "use_rot6d", False)),
            arm_mode=getattr(config, "ee_arm_mode", "right"),
            gripper_dims=int(getattr(config, "ee_gripper_dims", 1)),
        )
    ]

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        *contract_steps,
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=pi05_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
