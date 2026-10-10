"""Parallel inference over shared native parameters; packed projections and fused phases."""

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from WAMInfer._triton_attention import masked_attention
from WAMInfer._triton_inference import CROSS_RESIDUAL, FFN_RESIDUAL, SELF_INPUT, SELF_RESIDUAL, modulated_norm


def _parameter_signature(parameter):
    if parameter is None:
        return None
    version = object() if parameter.is_inference() else parameter._version
    return (
        id(parameter),
        parameter.data_ptr(),
        parameter.shape,
        parameter.stride(),
        parameter.dtype,
        parameter.device,
        version,
    )


def prepare_packed_weights(attention, names):
    """Refresh QKV or KV weights; return whether a captured graph is stale."""
    if attention.training or torch.is_grad_enabled():
        raise RuntimeError("Projection packing requires eval mode and disabled gradients.")
    projections = [getattr(attention, name) for name in names]
    prefix = f"_inference_{names}"
    signature = tuple(_parameter_signature(p) for layer in projections for p in (layer.weight, layer.bias))
    if signature == getattr(attention, f"{prefix}_signature", None):
        return False
    if len({(layer.in_features, layer.out_features) for layer in projections}) != 1:
        raise ValueError("Packed projections must have matching dimensions.")
    biases = [layer.bias for layer in projections]
    if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
        raise ValueError("Packed projections must all use bias or all omit it.")
    weight = torch.cat([layer.weight.detach() for layer in projections])
    bias = None if biases[0] is None else torch.cat([value.detach() for value in biases])
    attention.register_buffer(f"{prefix}_weight", weight, persistent=False)
    attention.register_buffer(f"{prefix}_bias", bias, persistent=False)
    setattr(attention, f"{prefix}_signature", signature)
    return True


def qk_rms_rope(q, k, norm_q, norm_k, frequencies, num_heads):
    """BF16 RMSNorm + RoPE for the fused runtime's native Wan/Action layers."""
    from WAMInfer._triton_inference import rms_rope

    return rms_rope(q, k, norm_q.weight, norm_k.weight, torch.view_as_real(frequencies), num_heads, norm_q.eps)


class View:
    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)


@dataclass
class ActionState:
    x_action: Tensor
    t_mod: Tensor
    action_freqs: Tensor
    context_mask: Tensor
    context_kv: tuple


@dataclass
class VideoState:
    hidden_states: Tensor
    time_mod: Tensor
    rope_freqs: Tensor
    context_mask: Tensor
    grid_frames: int
    grid_height: int
    grid_width: int
    extras: dict = field(default_factory=dict)


class ActionAdapter(View):
    def extract_prediction(self, state):
        return self.action_decoder(state.x_action)

    def prepare_state(self, noisy_actions, *, prepared_context, prepared_time):
        x = self.action_encoder(noisy_actions)
        (_, t_mod, index) = prepared_time
        t_mod = t_mod.index_select(0, index).squeeze(0)
        (_, context_mask, context_kv) = prepared_context
        return ActionState(
            x_action=x,
            t_mod=t_mod,
            action_freqs=self._get_rope_freqs(x.shape[1]).to(device=x.device),
            context_mask=context_mask,
            context_kv=context_kv,
        )

    def pre_attn_at_layer_for_compile(self, layer_id: int, astate: ActionState):
        """Compile-friendly pre-attention half using a tensor tuple post-state.

        Returns a 4-tuple ``(q, k, v, post_state)`` where ``post_state`` is the
        tensor tuple ``(residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
        """
        block = self.blocks[layer_id]
        chunks = (block.modulation.to(dtype=astate.t_mod.dtype, device=astate.t_mod.device) + astate.t_mod).chunk(
            6, dim=1
        )
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = chunks
        residual_x = astate.x_action
        attn_input = block.self_attn_norm(residual_x) * (1 + scale_msa) + shift_msa
        sa = block.self_attn
        q, k, v = F.linear(attn_input, sa._inference_qkv_weight, sa._inference_qkv_bias).chunk(3, dim=-1)
        q, k = qk_rms_rope(q, k, sa.norm_q, sa.norm_k, astate.action_freqs, self.num_heads)
        return (q, k, v, (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp))

    def post_attn_at_layer_for_compile(
        self, layer_id: int, astate: ActionState, attn_out: torch.Tensor, post_state: tuple[torch.Tensor, ...]
    ) -> ActionState:
        """Compile-friendly post-attention half consuming a tensor tuple."""
        block = self.blocks[layer_id]
        (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp) = post_state
        x = residual_x + gate_msa * block.self_attn.o(attn_out)
        text_mask = astate.context_mask
        if text_mask is not None:
            if text_mask.dim() == 2:
                text_mask = text_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        cross = block.cross_attn
        q = cross.norm_q(cross.q(block.context_attn_norm(x)))
        k, v = astate.context_kv[layer_id]
        mask = text_mask.unsqueeze(1) if text_mask is not None and text_mask.ndim == 3 else text_mask
        x = x + cross.o(masked_attention(q, k, v, mask, cross.num_heads, tail=True))
        mlp_input = block.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * block.ffn(mlp_input)
        astate.x_action = x
        return astate


class VideoAdapter(View):
    def prepare(self, **kw) -> VideoState:
        dit = self.dit
        latents = kw["latents"]
        prepared_time = kw.get("_inference_video_time")
        (time_embed, time_modulation, index) = prepared_time
        time_embed = time_embed.index_select(0, index).squeeze(0)
        time_modulation = time_modulation.index_select(0, index).squeeze(0)
        prepared_context = kw.get("_inference_video_context")
        (context, context_mask, _) = prepared_context
        hidden_states = latents
        if hidden_states.shape[0] != context.shape[0]:
            hidden_states = torch.concat([hidden_states] * context.shape[0], dim=0)
        hidden_states = dit.patch_embedding(hidden_states)
        (grid_frames, grid_height, grid_width) = hidden_states.shape[2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:grid_frames]
                    .view(grid_frames, 1, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[1][:grid_height]
                    .view(1, grid_height, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[2][:grid_width]
                    .view(1, 1, grid_width, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                ],
                dim=-1,
            )
            .reshape(grid_frames * grid_height * grid_width, 1, -1)
            .to(hidden_states.device)
        )
        extras = {"time_embed": time_embed}
        extras["_inference_context_kv"] = prepared_context[2]
        return VideoState(
            hidden_states=hidden_states,
            time_mod=time_modulation,
            rope_freqs=freqs,
            context_mask=context_mask,
            grid_frames=grid_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            extras=extras,
        )

    def pre_attn_at_layer_for_compile(self, layer_id, state):
        block = self.dit.blocks[layer_id]
        pending = state.extras.pop("_inference_ffn_residual", None)
        update, previous = (None, None) if pending is None else pending
        x, normalized = modulated_norm(
            state.hidden_states,
            update,
            block.modulation,
            previous,
            state.time_mod,
            None,
            None,
            SELF_INPUT if pending is None else FFN_RESIDUAL,
            block.norm1.eps,
        )
        sa = block.self_attn
        flat = normalized.reshape(-1, normalized.shape[-1])
        projected = torch.nn.functional.linear(flat, sa._inference_qkv_weight, sa._inference_qkv_bias)
        q, k, v = projected.reshape(*normalized.shape[:-1], projected.shape[-1]).chunk(3, dim=-1)
        q, k = qk_rms_rope(q, k, sa.norm_q, sa.norm_k, state.rope_freqs, sa.num_heads)
        state.hidden_states = x
        if layer_id == 1 and "token_reuse" in state.extras:
            from WAMInfer.approximation import select_tokens

            select_tokens(state, x)
        return q, k, v, (x,)

    def post_attn_at_layer_for_compile(self, layer_id, state, attended, fields):
        block = self.dit.blocks[layer_id]
        self_output = block.self_attn.o(attended)
        x, normalized = modulated_norm(
            fields[0],
            self_output,
            block.modulation,
            None,
            state.time_mod,
            block.norm3.weight,
            block.norm3.bias,
            SELF_RESIDUAL,
            block.norm3.eps,
        )
        mask = state.context_mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1).unsqueeze(1)
        cross = block.cross_attn
        k, v = state.extras["_inference_context_kv"][layer_id]
        query = cross.q(normalized)
        query = cross.norm_q(query)
        context_output = masked_attention(
            query, k, v, mask, cross.num_heads, tail=getattr(cross, "_inference_attention_tail", False)
        )
        cross_output = cross.o(context_output)
        x, normalized = modulated_norm(
            x, cross_output, block.modulation, None, state.time_mod, None, None, CROSS_RESIDUAL, block.norm2.eps
        )
        return normalized, x

    def ffn_at_layer_for_compile(self, layer_id, state, normalized, x):
        block = self.dit.blocks[layer_id]
        reuse = state.extras.get("token_reuse") if layer_id > 0 else None
        indices = state.extras.get("token_indices") if reuse is not None else None
        if indices is not None:
            normalized = normalized.index_select(1, indices)
        if self.ffn is not None:
            output = self.ffn.forward(layer_id, normalized)
        else:
            flat = normalized.reshape(-1, normalized.shape[-1])
            output = block.ffn[0](flat)
            output = block.ffn[1](output)
            output = block.ffn[2](output).reshape_as(normalized)
        if reuse is not None:
            n, count = reuse["n"], reuse["count"]
            if indices is not None:
                clean = reuse["cache"][layer_id - 1].index_copy(1, indices[:count], output[:, :count])
                output = torch.cat((clean, output[:, count:]), dim=1)
            state.extras["token_outputs"].append(output[:, :n].contiguous())
        if layer_id == self.num_layers - 1:
            modulation = block.modulation + state.time_mod
            gate = modulation[:, :, 5] if modulation.ndim == 4 else modulation[:, 5:6]
            state.hidden_states = x + gate * output
        else:
            state.hidden_states = x
            state.extras["_inference_ffn_residual"] = (output, block.modulation)
        return state

    def finalize(self, state: VideoState):
        """Wan DiT head + unpatchify. Returns ``(B, z_dim, F, H, W)``."""
        dit = self.dit
        head = dit.head
        time_embed = state.extras["time_embed"]
        head_time_embed = time_embed if time_embed.dim() == 3 else time_embed.unsqueeze(1)
        hidden_states = head(state.hidden_states, head_time_embed)
        hidden_states = dit.unpatchify(hidden_states, (state.grid_frames, state.grid_height, state.grid_width))
        return hidden_states

    def preprocess_input_for_inference(self, **kwargs):
        from WAMInfer.preparation import prepare_inputs

        return prepare_inputs(self, **kwargs)

    def decode_video(self, latents):
        return self._native.decode_video(latents, tiled=False)
