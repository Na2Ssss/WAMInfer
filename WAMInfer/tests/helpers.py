"""Small genuine upstream layers; no duplicate model definitions."""

from types import SimpleNamespace

import torch
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.model.video_backbone.wan.models.dit import WanModel


def make_arch(device="cpu", dtype=torch.float32, head_dim=8, mask="mutual"):
    torch.manual_seed(42)
    dit = WanModel(
        dim=4 * head_dim,
        in_dim=4,
        ffn_dim=8 * head_dim,
        out_dim=4,
        text_dim=24,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    arch = DualSystemSelfAttnArchitecture()
    arch.video_backbone = Wan22Ti2v(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))
    arch.video_backbone.text_encoder = None
    arch.action_backbone = ActionDiT(
        action_dim=7,
        dim=2 * head_dim,
        ffn_dim=4 * head_dim,
        num_heads=4,
        num_layers=2,
        video_dim=4 * head_dim,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        attn_head_dim=head_dim,
        text_dim=24,
    )
    arch._init_proprio_context(dict(use_proprioception=True, state_dim=7), text_dim=24)
    arch._mot_driver_kwargs = dict(attention_mask_mode=mask, video_attention_mask_mode="first_frame_causal")
    arch.build_mot_driver()
    arch.set_dtype_device(dtype, torch.device(device))
    return arch.eval()


def make_inputs(arch, seed=0):
    generator = torch.Generator(device=arch.device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=arch.device, dtype=arch.dtype)

    latents = randn(1, 4, 3, 4, 4)
    return dict(
        latents=latents,
        first_frame_latents=latents[:, :, :1].clone(),
        context=randn(1, 5, 24),
        context_mask=torch.tensor([[True, True, True, False, False]], device=arch.device),
        fuse_vae_embedding_in_latents=True,
        proprio=randn(1, 7),
    )
