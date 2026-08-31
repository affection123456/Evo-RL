from unittest.mock import patch

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.processor_pi05 import (
    make_pi05_pre_post_processors,
    pi05_raw32_batch_to_transition,
)
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
)
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_raw32_flow_noise_uses_full_model_action_shape() -> None:
    model = object.__new__(PI05Pytorch)
    noise = model.sample_noise((2, 50, 32), torch.device("cpu"))

    assert noise.shape == (2, 50, 32)
    assert torch.isfinite(noise).all()
    assert torch.count_nonzero(noise[..., 10:]) > 0


def _stats(dim: int) -> dict[str, torch.Tensor]:
    return {
        "count": torch.tensor(100),
        "min": torch.arange(dim, dtype=torch.float32),
        "max": torch.arange(dim, dtype=torch.float32) + 2,
        "mean": torch.arange(dim, dtype=torch.float32) + 1,
        "std": torch.full((dim,), 2.0),
    }


def test_raw32_converter_selects_ee_keys_and_builds_action_chunk() -> None:
    batch = {
        "ee_state": torch.randn(2, 44),
        "ee_actions": torch.randn(2, 50, 34),
        "state": torch.full((2, 44), 999.0),
        "actions": torch.full((2, 50, 34), 999.0),
        "top_head": torch.rand(2, 3, 16, 16),
        "hand_left": torch.rand(2, 3, 16, 16),
        "hand_right": torch.rand(2, 3, 16, 16),
        "task": ["pick", "pick"],
    }

    transition = pi05_raw32_batch_to_transition(batch)
    observation = transition[TransitionKey.OBSERVATION]

    assert set(observation) == {
        OBS_STATE,
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    }
    torch.testing.assert_close(observation[OBS_STATE], batch["ee_state"][..., :32])
    torch.testing.assert_close(transition[TransitionKey.ACTION], batch["ee_actions"][..., :32])
    assert transition[TransitionKey.ACTION].shape == (2, 50, 32)


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_raw32_processor_checkpoint_round_trip(mock_tokenizer, tmp_path) -> None:
    cfg = PI05Config(device="cpu", dtype="float32")
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
    }
    stats = {OBS_STATE: _stats(32), ACTION: _stats(32)}
    preprocessor, postprocessor = make_pi05_pre_post_processors(cfg, stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
        to_transition=pi05_raw32_batch_to_transition,
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_postprocessor.json",
    )
    loaded_normalizer = next(
        step for step in loaded_preprocessor.steps if isinstance(step, NormalizerProcessorStep)
    )
    loaded_unnormalizer = next(
        step
        for step in loaded_postprocessor.steps
        if isinstance(step, UnnormalizerProcessorStep)
    )

    assert set(loaded_normalizer._tensor_stats) == {OBS_STATE, ACTION}
    assert set(loaded_unnormalizer._tensor_stats) == {OBS_STATE, ACTION}
    assert loaded_normalizer.features[OBS_STATE].shape == (32,)
    assert loaded_unnormalizer.features[ACTION].shape == (32,)
    assert loaded_normalizer.norm_map[FeatureType.STATE] is NormalizationMode.MIN_MAX
    assert loaded_unnormalizer.norm_map[FeatureType.ACTION] is NormalizationMode.MEAN_STD
