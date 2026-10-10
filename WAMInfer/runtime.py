"""Parallel inference with two optional approximations; upstream weights stay shared."""

import threading
from functools import partial
import torch
from openwam.deploy.denoise_schedule import schedule_sync
from openwam.deploy.model_loader import load_from_checkpoint_dir
from WAMInfer.blocks import View, VideoAdapter, ActionAdapter, prepare_packed_weights
from WAMInfer.ffn import PreparedFFN
from WAMInfer.graphs import CudaGraphForward, TextEncodeGraph, VAEEncodeGraph, module_guard
from WAMInfer.joint import JointDriver, ParallelJointLoop
from WAMInfer.preparation import ConditioningPreparation, FirstFramePreparation
from WAMInfer.approximation import TokenReuse, MotionRefinement


class Runtime(View):
    def __init__(
        self,
        architecture,
        *,
        reuse_tokens=False,
        adaptive_2f4f=False,
        refresh_buckets=(32, 64, 96, 120),
        motion_threshold=0.08,
    ):
        super().__init__(architecture)
        if (
            architecture.device.type != "cuda"
            or architecture.dtype != torch.bfloat16
            or architecture.mot_driver.head_dim != 128
        ):
            raise ValueError("Parallel requires CUDA BF16 and head_dim=128")
        if (
            architecture.video_backbone.__class__.__name__ != "Wan22Ti2v"
            or architecture.action_backbone.variant != "joint_self_attn"
            or architecture.video_backbone.external_encoder is not None
        ):
            raise ValueError("Parallel requires Wan22Ti2v joint_self_attn with the native VAE")
        self._lock = threading.Lock()
        self.closed = False
        self.video_backbone = VideoAdapter(architecture.video_backbone)
        self.action_backbone = ActionAdapter(architecture.action_backbone)
        self.mot_driver = JointDriver(architecture.mot_driver, self.video_backbone, self.action_backbone)
        vb = self.video_backbone
        vb.ffn = PreparedFFN(vb) if torch.cuda.get_device_capability(self.device) == (8, 9) else None
        vb.input_stream = torch.cuda.Stream(device=self.device)
        vb.text_graph = TextEncodeGraph(vb.text_encoder) if vb.text_encoder is not None else None
        vb.vae_graph = VAEEncodeGraph(vb.vae) if vb.vae is not None else None
        self._cuda_graph_forward = CudaGraphForward(self._forward_impl, reuse_unchanged_inputs=True)
        self._action_cuda_graph_forward = CudaGraphForward(
            partial(self._forward_impl, _inference_action_only=True), shared_inputs=self._cuda_graph_forward
        )
        self._compiled_mot_run_joint_loop = ParallelJointLoop(self)
        self._first_frame_preparation = FirstFramePreparation(self)
        self._conditioning_preparation = ConditioningPreparation(self)
        self._inference_eval_guard = self._inference_module_guard = None
        self.token_reuse = TokenReuse(refresh_buckets) if reuse_tokens else None
        self.adaptive_2f4f, self.motion_threshold = adaptive_2f4f, motion_threshold
        self._residual_graphs = {}
        self.last_stats = {}

    def reset(self):
        """Start a new episode/camera stream without discarding warmed graphs."""
        with self._lock:
            if self.token_reuse is not None:
                self.token_reuse.reset()

    def generate(self, *args, **kwargs):
        with self._lock:
            if self.closed:
                raise RuntimeError("This runtime is closed")
            return self._generate(*args, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        schedule,
        prompt: str,
        *,
        first_frame_image=None,
        num_frames: int = 9,
        action_num_frames: int | None = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        decode_video: bool = False,
        proprio: torch.Tensor | None = None,
        active_action_mask: torch.Tensor | None = None,
        motion_metric=None,
    ) -> dict:
        """Run the original sampler; optional switches reduce Transformer/FFN work."""
        import math

        refinement = MotionRefinement(motion_metric, self.motion_threshold, schedule) if self.adaptive_2f4f else None
        eval_guard = self._inference_eval_guard
        if eval_guard is None or not eval_guard.all_eval():
            self.eval()
            self._inference_eval_guard = module_guard(self)
        if len(schedule) < 2:
            raise ValueError("Schedule requires at least one step.")
        if not all((len(pair) == 2 and all((math.isfinite(float(t)) for t in pair)) for pair in schedule)):
            raise ValueError("Schedule must contain finite video/action timestep pairs.")
        if tuple(schedule[-1]) != (0.0, 0.0):
            raise ValueError("Schedule must terminate at (0, 0).")
        if any(
            (not (v > vn and a > an and (vn >= 0) and (an >= 0)) for ((v, a), (vn, an)) in zip(schedule, schedule[1:]))
        ):
            raise ValueError("Every step must advance both modalities; plateau/skip schedules are unsupported.")
        (vb, device, dtype) = (self.video_backbone, self.device, self.dtype)
        action_num_frames = int(num_frames if action_num_frames is None else action_num_frames)
        if action_num_frames < 2:
            raise ValueError("action_num_frames must be >= 2.")
        inputs = vb.preprocess_input_for_inference(
            prompt=prompt,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        input_stream = inputs.pop("_inference_input_stream", None)
        reference = inputs.get("first_frame_latents")
        if self.uses_proprioception:
            if proprio is None:
                raise ValueError("This checkpoint requires proprioception.")
            inputs["proprio"] = proprio.to(device=device, dtype=dtype)
        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        inactive = self._resolve_inactive_action_dims(active_action_mask, "cpu")
        if inactive is not None:
            inactive = inactive.nonzero(as_tuple=True)[0].to(device, non_blocking=True)
        inactive_noise = None
        inputs = self.prepare_inference_inputs(inputs)
        reuse = observation = None
        if self.token_reuse is not None:
            reuse, observation = self.token_reuse.begin(first_frame_image, inputs, prompt, vb._dit_patch_size)
        conditioning = self._conditioning_preparation
        inputs = conditioning.prepare(inputs, schedule)
        if input_stream is not None:
            main = torch.cuda.current_stream(device)
            main.wait_stream(input_stream)
            reference.record_stream(main)
        if reference is not None:
            latents = inputs["latents"].clone()
            latents[:, :, : reference.shape[2]] = reference
            inputs["latents"] = latents
        first_frame = self._first_frame_preparation if inputs["latents"].shape[2] > 1 else None
        train_v = float(self.video_scheduler.num_train_timesteps)
        train_a = float(self.action_scheduler.num_train_timesteps)
        full_steps = []
        for index, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(schedule, schedule[1:])):
            torch.compiler.cudagraph_mark_step_begin()
            forward_inputs = conditioning.step_inputs(inputs, index)
            fresh = refinement is None or refinement.full(index)
            if fresh:
                full_steps.append(index)
            if refinement is not None:
                if fresh:
                    forward_inputs["_inference_record_residual"] = index != 9 or decode_video
                else:
                    forward_inputs["_inference_residual"] = refinement.residual
            if first_frame is not None and index == 0:
                if reuse is not None:
                    forward_inputs["_inference_token_reuse"] = reuse
                result = first_frame.run_first(action_latents, **forward_inputs)
                inputs["_inference_first_frame"] = result[2]
                if reuse is not None:
                    features, cache = result[3:5]
            else:
                if not decode_video and index == len(schedule) - 2:
                    forward_inputs["_inference_action_only"] = True
                result = self.forward(action_latents, **forward_inputs)
            video_prediction, action_prediction = result[:2]
            if refinement is not None and fresh:
                if index != 9:
                    refinement.residual = result[-1]
                if index == 3:
                    clean = action_latents.float() - (ta / train_a) * action_prediction.float()
                    if inactive is not None:
                        clean[..., inactive] = 0
                    refinement.decide(self._decode_actions(clean))
            if video_prediction is not None:
                inputs["latents"] = inputs["latents"] + video_prediction * ((tv_next - tv) / train_v)
                if reference is not None:
                    inputs["latents"] = inputs["latents"].clone()
                    inputs["latents"][:, :, : reference.shape[2]] = reference
            if action_prediction is None:
                raise RuntimeError("The joint model must produce fresh Action predictions at every step.")
            if inactive is not None and inactive_noise is None:
                inactive_noise = action_latents[..., inactive].detach().clone() / (ta / train_a)
            action_latents = self.action_scheduler.flow_step(
                action_prediction, ta / train_a, ta_next / train_a, action_latents
            )
            if inactive is not None:
                action_latents[..., inactive] = inactive_noise * (ta_next / train_a)
        video = vb.decode_video(inputs["latents"]) if decode_video else None
        actions = self._decode_actions(action_latents)
        if reuse is not None:
            self.token_reuse.commit(observation, features, cache)
        self.last_stats = dict(
            full_evaluations=tuple(full_steps),
            refreshed_tokens=None if reuse is None else reuse["count"],
            motion_m=None if refinement is None else refinement.amplitude,
        )
        return {"video": video, "actions": actions}

    def _decode_actions(self, latents):
        actions = latents.squeeze(0).float().cpu().numpy()
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize(actions)
        return actions

    def forward(self, noisy_actions, **pipeline_inputs):
        if "_inference_residual" in pipeline_inputs:
            key = pipeline_inputs.get("_inference_action_only", False)
            if key not in self._residual_graphs:
                self._residual_graphs[key] = CudaGraphForward(self._forward_impl, reuse_unchanged_inputs=True)
            return self._residual_graphs[key](noisy_actions, **pipeline_inputs)
        graph = (
            self._action_cuda_graph_forward
            if pipeline_inputs.pop("_inference_action_only", False)
            else self._cuda_graph_forward
        )
        return graph(noisy_actions, **pipeline_inputs)

    def _forward_impl(self, noisy_actions, **inputs):
        (vb, ab) = (self.video_backbone, self.action_backbone)
        vstate = vb.prepare(**inputs)
        if inputs.get("_inference_first_frame") is not None:
            vstate.extras["first_frame_kv"] = inputs["_inference_first_frame"]
        vstate.extras["record_first_frame"] = bool(inputs.get("_inference_record_first_frame", False))
        vstate.extras["action_only"] = bool(inputs.get("_inference_action_only", False))
        astate = ab.prepare_state(
            noisy_actions,
            prepared_context=inputs["_inference_action_context"],
            prepared_time=inputs["_inference_action_time"],
        )
        reuse = inputs.get("_inference_token_reuse")
        if reuse is not None:
            vstate.extras.update(token_reuse=reuse, token_outputs=[])
        before_v, before_a = vstate.hidden_states, astate.x_action
        prefix = inputs.get("_inference_first_frame")
        clean = prefix[1].shape[1] if prefix is not None else 0
        residual = inputs.get("_inference_residual")
        if residual is None:
            (vstate, astate) = self._compiled_mot_run_joint_loop(vstate, astate)
        else:
            future = before_v[:, clean:] + residual[0]
            vstate.hidden_states = torch.cat((prefix[1], future), dim=1) if prefix is not None else future
            astate.x_action = before_a + residual[1]
        result = (None if vstate.extras["action_only"] else vb.finalize(vstate), ab.extract_prediction(astate))
        if vstate.extras["record_first_frame"]:
            bank = vstate.extras["first_frame_result"]
            clean = bank[1].shape[1]
            result += (bank,)
        if reuse is not None:
            result += (vstate.extras["token_features"], tuple(vstate.extras["token_outputs"]))
        if inputs.get("_inference_record_residual", False):
            result += ((vstate.hidden_states[:, clean:] - before_v[:, clean:], astate.x_action - before_a),)
        return result

    def prepare_inference_inputs(self, inputs):
        vb, ab = self.video_backbone, self.action_backbone
        latents = inputs["latents"]
        rows = (
            latents.shape[0]
            * latents.shape[2]
            * latents.shape[-2]
            * latents.shape[-1]
            // (vb._dit_patch_size[1] * vb._dit_patch_size[2]),
        )
        if latents.shape[2] > 1:
            rows += (rows[0] * (latents.shape[2] - 1) // latents.shape[2],)
        key = (rows, self.mot_driver.attention_mask_mode, vb.video_attention_mask_mode)
        guard = self._inference_module_guard
        if guard is not None and key == self._inference_preparation_key and not guard.changed():
            return inputs
        for blocks in (vb.dit.blocks, ab.blocks):
            for block in blocks:
                prepare_packed_weights(block.cross_attn, "kv")
                prepare_packed_weights(block.self_attn, "qkv")
        if vb.ffn is not None:
            if self.token_reuse is not None:
                n = rows[0] // latents.shape[2]
                rows += tuple(rows[0] - n + b for b in self.token_reuse.capacities(n))
            vb.ffn.prepare(rows)
        self._cuda_graph_forward.reset()
        self._action_cuda_graph_forward.reset()
        self._first_frame_preparation.close()
        self._residual_graphs.clear()
        if self.token_reuse is not None:
            self.token_reuse.reset()
        vb.dit.freqs = tuple(freq.to(device=self.device) for freq in vb.dit.freqs)
        ab._sync_rope_freqs_device()
        self._inference_module_guard = module_guard(self, track_versions=True)
        self._inference_preparation_key = key
        return inputs

    def close(self):
        with self._lock:
            if self.closed:
                return self._native
            torch.cuda.synchronize(self.device)
            self._cuda_graph_forward.reset()
            self._action_cuda_graph_forward.reset()
            self._first_frame_preparation.close()
            self._residual_graphs.clear()
            if self.token_reuse is not None:
                self.token_reuse.reset()
            self._conditioning_preparation.close()
            vb = self.video_backbone
            for graph in (vb.text_graph, vb.vae_graph):
                if graph is not None:
                    graph.close()
            if vb.ffn is not None:
                vb.ffn.close()
            for blocks in (vb.dit.blocks, self.action_backbone.blocks):
                for block in blocks:
                    for attention, names in ((block.self_attn, "qkv"), (block.cross_attn, "kv")):
                        for suffix in ("weight", "bias", "signature"):
                            key = f"_inference_{names}_{suffix}"
                            if hasattr(attention, key):
                                delattr(attention, key)
            self.closed = True
            return self._native


class OpenWAM:
    """Checkpoint convenience API; the original architecture remains accessible."""

    def __init__(
        self,
        checkpoint_dir,
        *,
        device="cuda",
        checkpoint_name=None,
        reuse_tokens=False,
        adaptive_2f4f=False,
        refresh_buckets=(32, 64, 96, 120),
        motion_threshold=0.08,
    ):
        self.config, self.original = load_from_checkpoint_dir(checkpoint_dir, device=device, ckpt_name=checkpoint_name)
        self.architecture = Runtime(
            self.original,
            reuse_tokens=reuse_tokens,
            adaptive_2f4f=adaptive_2f4f,
            refresh_buckets=refresh_buckets,
            motion_threshold=motion_threshold,
        )
        self._lock = threading.Lock()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        image,
        *,
        proprio=None,
        seed=42,
        num_inference_steps=10,
        num_frames=9,
        action_num_frames=33,
        height=384,
        width=320,
        decode_video=False,
        active_action_mask=None,
        motion_metric=None,
    ):
        with self._lock:
            arch = self.original
            schedule = schedule_sync(
                arch.video_scheduler,
                arch.action_scheduler,
                num_steps=num_inference_steps,
                shift=float(arch.action_backbone.shift_action or 5.0),
                shift_video=float(arch.video_backbone.shift_video or 5.0),
            )
            kwargs = dict(
                first_frame_image=image,
                proprio=arch.normalize_deploy_proprio(proprio),
                seed=seed,
                num_frames=num_frames,
                action_num_frames=action_num_frames,
                height=height,
                width=width,
                decode_video=decode_video,
                active_action_mask=active_action_mask,
                motion_metric=motion_metric,
            )
            return self.architecture.generate(schedule, prompt, **kwargs)

    def reset(self):
        with self._lock:
            self.architecture.reset()

    def close(self):
        with self._lock:
            self.architecture.close()
