#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Policy,
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from lerobot.policies.pi0_dmp.configuration_pi0_dmp import PI0DMPConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

if TYPE_CHECKING:
    from lerobot.policies.pi0.modeling_pi0 import ActionSelectKwargs


class PI0DMPPytorch(PI0Pytorch):
    """PI0 core model variant that can condition on reference state/actions."""

    def embed_suffix(self, state, noisy_actions, timestep, ref_state=None):
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)
            if ref_state is not None:
                ref_state = ref_state.to(torch.float32)

        if ref_state is None:
            ref_state = state

        ref_state_emb = self._apply_checkpoint(self.state_proj, ref_state)
        state_emb = self._apply_checkpoint(self.state_proj, state)

        embs.append(ref_state_emb[:, None, :])
        embs.append(state_emb[:, None, :])

        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        pad_masks.append(state_mask)
        att_masks += [1, 1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_tokens):
            x = self.action_time_mlp_in(action_time_tokens)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        ref_state=None,
        ref_actions=None,
    ) -> Tensor:
        """Training forward pass with optional reference trajectory as flow noise."""
        if noise is None:
            noise = ref_actions if ref_actions is not None else self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time, ref_state=ref_state
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_tokens, suffix_tokens, attn_mask, pos_ids, ada_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_tokens, suffix_tokens],
                use_cache=False,
                adarms_cond=[None, ada_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        ref_state=None,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        if ref_state is None:
            ref_state = state

        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, time_tensor, ref_state=ref_state
            )

            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
        return x_t


class PI0DMPPolicy(PI0Policy):
    """PI0-DMP policy with reference-conditioned flow."""

    config_class = PI0DMPConfig
    name = "pi0_dmp"

    def __init__(self, config: PI0DMPConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        # Replace the parent PI0 core model with the DMP-capable variant.
        self.model = PI0DMPPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @staticmethod
    def _expand_or_skip_tensor(source: Tensor, target: Tensor) -> tuple[Tensor | None, bool]:
        if source.shape == target.shape:
            return source, False
        if source.ndim != target.ndim:
            return None, False

        expanded = target.clone()
        slices = tuple(slice(0, min(src_dim, dst_dim)) for src_dim, dst_dim in zip(source.shape, target.shape))
        if any(s.stop == 0 for s in slices):
            return None, False
        expanded[slices] = source[slices].to(dtype=expanded.dtype, device=expanded.device)
        return expanded, True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> PI0DMPPolicy:
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, PI0DMPConfig):
            pi0_dmp_fields = {field.name for field in fields(PI0DMPConfig)}
            config = PI0DMPConfig(**{k: v for k, v in asdict(config).items() if k in pi0_dmp_fields})

        model = cls(config, **kwargs)
        ckpt_path = Path(pretrained_name_or_path)
        if ckpt_path.is_dir():
            resolved_file = ckpt_path / "model.safetensors"
        else:
            from transformers.utils import cached_file

            resolved_file = Path(
                cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )

        if not resolved_file.is_file():
            raise FileNotFoundError(f"model.safetensors not found at {resolved_file}")

        original_state_dict = load_file(str(resolved_file))
        fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
        remapped_state_dict = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed_state_dict.items()
        }

        target_state = model.state_dict()
        loadable_state: dict[str, Tensor] = {}
        expanded_keys: list[str] = []
        skipped_shape_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        unexpected_keys: list[str] = []

        for key, value in remapped_state_dict.items():
            if key not in target_state:
                unexpected_keys.append(key)
                continue
            adapted, expanded = cls._expand_or_skip_tensor(value, target_state[key])
            if adapted is None:
                skipped_shape_keys.append((key, tuple(value.shape), tuple(target_state[key].shape)))
                continue
            loadable_state[key] = adapted
            if expanded:
                expanded_keys.append(key)

        missing_keys, load_unexpected_keys = model.load_state_dict(loadable_state, strict=strict)
        print(
            "PI0-DMP partial checkpoint load | "
            f"loaded={len(loadable_state)} expanded={len(expanded_keys)} "
            f"skipped_shape={len(skipped_shape_keys)} unexpected={len(unexpected_keys) + len(load_unexpected_keys)} "
            f"missing={len(missing_keys)}"
        )
        if expanded_keys:
            print("Expanded checkpoint tensors:", expanded_keys[:10])
        if skipped_shape_keys:
            print("Skipped incompatible tensors:", skipped_shape_keys[:10])

        model.to(config.device)
        model.eval()
        return model

    def prepare_ref_state(self, batch: dict[str, Tensor]) -> Tensor:
        key = self.config.ref_state_key
        if key in batch:
            return pad_vector(batch[key], self.config.max_state_dim)
        return self.prepare_state(batch)

    def prepare_ref_action(self, batch: dict[str, Tensor]) -> Tensor | None:
        key = self.config.ref_action_key
        if key not in batch:
            return None
        return pad_vector(batch[key], self.config.max_action_dim)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        ref_state = self.prepare_ref_state(batch)
        ref_actions = self.prepare_ref_action(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            ref_state=ref_state,
            noise=ref_actions,
            **kwargs,
        )
        return actions[:, :, : self.config.max_action_dim]

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        images = []
        img_masks = []
        device = next(self.parameters()).device
        image_keys = [
            "observation.images.top_head",
            "observation.images.hand_right",
            *self.config.ref_image_features,
        ]

        for key in image_keys:
            if key not in batch:
                raise KeyError(f"PI0-DMP requires image feature '{key}' in the batch.")
            img = batch[key].to(device=device, dtype=torch.float32)
            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

        return images, img_masks

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        ref_state = self.prepare_ref_state(batch)
        ref_actions = self.prepare_ref_action(batch)

        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            ref_state=ref_state,
            ref_actions=ref_actions,
        )

        per_sample_loss = losses.mean(dim=(1, 2))

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            loss_dict["loss"] = float(per_sample_loss.mean().item())
            return per_sample_loss, loss_dict

        loss = per_sample_loss.mean()
        loss_dict["loss"] = float(loss.item())
        return loss, loss_dict
