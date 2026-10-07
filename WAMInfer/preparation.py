"""Exact prompt caching and request-local context, timestep and current-frame preparation."""

import torch
from openwam.model.video_backbone.wan.dit_forward import build_time_modulation as native_time_modulation
from openwam.model.video_backbone.wan.preprocess import check_resize_height_width, preprocess_video
from WAMInfer.graphs import CudaGraphForward, module_guard


def build_time_modulation(dit, timestep, latents, *, patch_size, fuse_vae_embedding_in_latents):
    return native_time_modulation(
        dit,
        timestep,
        latents,
        patch_size=patch_size,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        force_per_token_t_mod=True,
        num_clean_prefix_frames=0,
        zero_clean_prefix_t_mod=True,
        has_first_frame_latents=fuse_vae_embedding_in_latents,
    )


def encode_text(prompt, *, tokenizer, graph, device):
    ids, mask = tokenizer(
        [prompt], return_mask=True, add_special_tokens=True, max_length=512, padding="max_length", truncation=True
    )
    length = int(mask.gt(0).sum(dim=1).max())
    bucket = next(size for size in (32, 64, 128, 256, 512) if size >= length)
    context, lengths = graph(ids.to(device), mask.to(device))
    return context[:, :bucket].contiguous(), lengths


def prepare_inputs(video, *, prompt, first_frame_image, num_frames, height, width, seed):
    height, width, num_frames = check_resize_height_width(
        height,
        width,
        num_frames,
        height_division_factor=video._height_division_factor,
        width_division_factor=video._width_division_factor,
        time_division_factor=video._time_division_factor,
        time_division_remainder=video._time_division_remainder,
    )

    def encode_image():
        frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        pixels = preprocess_video(frames, dtype=video.dtype, device=video.device)
        return video.vae_graph(pixels).to(dtype=video.dtype, device=video.device)

    reference = None
    side = video.input_stream
    if side is not None and first_frame_image is not None:
        side.wait_stream(torch.cuda.current_stream(video.device))
        with torch.cuda.stream(side):
            reference = encode_image()
    context, lengths = video.text_graph.encode_prompt(prompt, video._tokenizer, video.device)
    scale = video.vae.upsampling_factor
    shape = (1, video.vae.model.z_dim, (num_frames - 1) // 4 + 1, height // scale, width // scale)
    generator = None if seed is None else torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        dtype=video.dtype, device=video.device
    )
    inputs = dict(
        context=context, seq_lens=lengths, latents=latents, fuse_vae_embedding_in_latents=first_frame_image is not None
    )
    if first_frame_image is not None:
        if reference is None:
            reference = encode_image()
        else:
            inputs["_inference_input_stream"] = side
        inputs["first_frame_latents"] = reference
    return inputs


class ConditioningPreparation:
    def __init__(self, architecture):
        self.architecture = architecture
        projection = torch.compile(self._project, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        self.context_graph = CudaGraphForward(projection)
        self.guard = None
        self.time_key = None
        self.time_values = None
        self.time_indexes = None
        self.time_builds = 0

    def close(self):
        self.context_graph.reset()
        self.guard = None
        self.time_key = self.time_values = None
        self.time_indexes = None

    def _project(self, context, context_mask, seq_lens, proprio):
        arch = self.architecture
        prepared = arch._append_proprio_context_token(
            dict(context=context, context_mask=context_mask, seq_lens=seq_lens), proprio
        )
        context = prepared["context"]
        mask = prepared.get("context_mask")
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
            if prepared.get("seq_lens") is not None:
                mask = torch.arange(context.shape[1], device=context.device)[None] < prepared["seq_lens"][:, None]

        def project(blocks, embedding):
            values = []
            for block in blocks:
                cross = block.cross_attn
                k, v = torch.nn.functional.linear(
                    embedding, cross._inference_kv_weight, cross._inference_kv_bias
                ).chunk(2, dim=-1)
                values.append((cross.norm_k(k), v))
            return (embedding, mask, tuple(values))

        (vb, ab) = (arch.video_backbone, arch.action_backbone)
        return (
            project(vb.dit.blocks, vb.dit.text_embedding(context)),
            project(ab.blocks, ab.text_embedding(context.to(device=arch.device, dtype=arch.dtype))),
        )

    @torch.no_grad()
    def prepare(self, inputs, schedule):
        arch = self.architecture
        if arch.training:
            raise ValueError("Conditioning preparation requires eval TI2V inference.")
        if self.guard is None:
            self.guard = module_guard(arch, track_versions=True)
        if self.guard.changed():
            self.context_graph.reset()
            self.time_key = self.time_values = None
        result = dict(inputs)
        (video, action) = self.context_graph(
            inputs["context"], inputs.get("context_mask"), inputs.get("seq_lens"), inputs.get("proprio")
        )
        result["_inference_video_context"] = video
        result["_inference_action_context"] = action
        self._prepare_time(inputs, schedule)
        return result

    def _prepare_time(self, inputs, schedule):
        arch = self.architecture
        (vb, ab) = (arch.video_backbone, arch.action_backbone)
        latents = inputs["latents"]
        key = (
            tuple((tuple((float(t) for t in pair)) for pair in schedule[:-1])),
            tuple(latents.shape),
            latents.device,
            latents.dtype,
            tuple(vb._dit_patch_size),
            bool(inputs.get("fuse_vae_embedding_in_latents", False)),
        )
        if key == self.time_key:
            return
        values = []
        for video_t, action_t in key[0]:
            vt = torch.full((latents.shape[0],), video_t, device=latents.device, dtype=latents.dtype)
            at = torch.full((latents.shape[0],), action_t, device=latents.device, dtype=latents.dtype)
            video = build_time_modulation(
                vb.dit, vt, latents, patch_size=vb._dit_patch_size, fuse_vae_embedding_in_latents=key[5]
            )
            embed = ab.time_embedding(at)
            values.append((video, (embed, ab.time_projection(embed))))
        self.time_values = tuple(
            (
                tuple((torch.stack([value[branch][component] for value in values]) for component in range(2)))
                for branch in range(2)
            )
        )
        self.time_indexes = torch.arange(len(values), device=latents.device)
        self.time_key = key
        self.time_builds += 1

    def step_inputs(self, inputs, index):
        (video, action) = self.time_values
        step = self.time_indexes[index : index + 1]
        (video, action) = ((*video, step), (*action, step))
        return dict(inputs, _inference_video_time=video, _inference_action_time=action)


class FirstFramePreparation:
    def __init__(self, architecture):
        self.architecture = architecture
        self.prefills = 0
        self.graph = CudaGraphForward(architecture._forward_impl, reuse_unchanged_inputs=True)

    def close(self):
        self.graph.reset()

    def run_first(self, noisy_actions, **inputs):
        arch = self.architecture
        vb = arch.video_backbone
        if arch.training or torch.is_grad_enabled():
            raise RuntimeError("First-frame preparation requires eval mode and disabled gradients.")
        if vb.video_attention_mask_mode != "first_frame_causal":
            raise ValueError("First-frame preparation requires first_frame_causal attention.")
        if not inputs.get("fuse_vae_embedding_in_latents", False):
            raise ValueError("First-frame preparation requires zero-timestep TI2V conditioning.")
        first = inputs.get("first_frame_latents")
        if first is None or first.ndim != 5 or first.shape[2] != 1 or (vb._dit_patch_size[0] != 1):
            raise ValueError("First-frame preparation requires exactly one clean latent frame.")
        if "_inference_first_frame" in inputs or "_inference_record_first_frame" in inputs:
            raise ValueError("The first evaluation must use the current observation without an existing bank.")
        result = self.graph(noisy_actions, **inputs, _inference_record_first_frame=True)
        self.prefills += 1
        return result
