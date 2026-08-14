from types import SimpleNamespace

import torch

from lerobot.scripts import lerobot_policy_infer_openpi_bridge as bridge_module


class _FakePaligemmaWithExpert:
    def __init__(self) -> None:
        weight = torch.zeros(1, dtype=torch.float32)
        q_proj = SimpleNamespace(weight=weight)
        self_attn = SimpleNamespace(q_proj=q_proj)
        layer = SimpleNamespace(self_attn=self_attn)
        language_model = SimpleNamespace(
            layers=[layer],
            config=SimpleNamespace(_attn_implementation=None),
        )
        self.paligemma = SimpleNamespace(language_model=language_model)

    def forward(self, **kwargs):
        del kwargs
        return None, "cached-prefix"


class _FakePI0DMPModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_inference_steps=2,
            chunk_size=1,
            max_action_dim=4,
        )
        self.paligemma_with_expert = _FakePaligemmaWithExpert()

    def _mask_padded_action_dims(self, actions: torch.Tensor) -> torch.Tensor:
        masked = actions.clone()
        masked[..., 2:] = 0
        return masked

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        del images, img_masks, lang_tokens, lang_masks
        return (
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        )

    def _prepare_attention_masks_4d(self, masks: torch.Tensor) -> torch.Tensor:
        return masks


def test_pi0_dmp_cached_sampler_masks_initial_noise_and_every_update(monkeypatch) -> None:
    denoise_inputs: list[torch.Tensor] = []

    def fake_denoise_step(model, **kwargs):
        del model
        denoise_inputs.append(kwargs["x_t"].clone())
        return torch.ones_like(kwargs["x_t"])

    monkeypatch.setattr(bridge_module, "_pi0_dmp_denoise_step_cached", fake_denoise_step)
    model = _FakePI0DMPModel()
    noise = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    result = bridge_module._pi0_dmp_sample_actions_cached(
        model,
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(1, 1, dtype=torch.long),
        lang_masks=torch.ones(1, 1, dtype=torch.bool),
        state=torch.zeros(1, 4),
        noise=noise,
    )

    assert len(denoise_inputs) == 2
    for denoise_input in denoise_inputs:
        torch.testing.assert_close(denoise_input[..., 2:], torch.zeros_like(denoise_input[..., 2:]))
    torch.testing.assert_close(result, torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]))
