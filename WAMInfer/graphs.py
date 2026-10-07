"""CUDA Graph replay, native guards, text caching and VAE input execution."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from contextlib import contextmanager
from types import MethodType
import torch.nn.functional as F
from openwam.model.video_backbone.wan.models.vae import CausalConv3d


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="wam_infer_guards_v1",
        sources=[str(Path(__file__).with_name("_inference_guards.cpp"))],
        extra_cflags=["-O3"],
        verbose=False,
    )


def module_guard(module, *, track_versions=False):
    return extension().ModuleGuard(getattr(module, "_native", module), track_versions)


class CudaGraphForward:
    """A bounded graph; serialized variants may share input storage and a pool."""

    def __init__(self, forward, *, reuse_unchanged_inputs=False, shared_inputs=None):
        self.forward = forward
        self.shared_inputs = shared_inputs
        self.reuse_unchanged_inputs = shared_inputs.reuse_unchanged_inputs if shared_inputs else reuse_unchanged_inputs
        self.captures = 0
        self.replays = 0
        self.fast_replays = 0
        self.reset()

    def reset(self):
        self.graph = None
        self.signature = None
        self.inputs = None
        self.outputs = None
        self.input_guard = None
        self.sources = self.versions = None

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        shared = self.shared_inputs
        if shared is not None and shared.signature == self.signature and shared.inputs is not self.inputs:
            self.reset()
        if self.input_guard is not None:
            leaves = self.input_guard.flatten((args, kwargs))
            if leaves is not None:
                self.fast_replays += 1
                return self._replay(leaves)
        (leaves, spec) = tree_flatten((args, kwargs))
        signature = []
        devices = set()
        for value in leaves:
            if isinstance(value, torch.Tensor):
                if value.device.type != "cuda":
                    raise ValueError("CUDA Graph forward requires device-resident tensor inputs.")
                devices.add(value.device)
                signature.append((value.shape, value.stride(), value.dtype, value.device))
            elif value is None or isinstance(value, (bool, int, float, str)):
                signature.append((type(value), value))
            else:
                raise TypeError(f"Unsupported CUDA Graph input type: {type(value).__name__}")
        if len(devices) != 1:
            raise ValueError("CUDA Graph forward requires tensors on exactly one CUDA device.")
        signature = (spec, tuple(signature))
        if signature != self.signature:
            self.reset()
            if shared is not None and shared.signature == signature:
                # Sequential variants share both storage and version tracking:
                # either graph's copies must invalidate the other's old values.
                self.inputs, self.sources, self.versions = shared.inputs, shared.sources, shared.versions
            else:
                self.inputs = [value.clone() if isinstance(value, torch.Tensor) else value for value in leaves]
                self.sources = [None] * len(leaves)
                self.versions = [None] * len(leaves)
            (static_args, static_kwargs) = tree_unflatten(self.inputs, spec)
            device = next(iter(devices))
            with torch.cuda.device(device):
                current = torch.cuda.current_stream(device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(*static_args, **static_kwargs)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                pool = shared.graph.pool() if shared is not None and self.inputs is shared.inputs else None
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    outputs = self.forward(*static_args, **static_kwargs)
                current.wait_stream(stream)
            (self.graph, self.outputs) = (graph, outputs)
            self.signature = signature
            self.captures += 1
        self.input_guard = extension().InputGuard((args, kwargs))
        return self._replay(leaves)

    def _replay(self, leaves):
        for index, (destination, source) in enumerate(zip(self.inputs, leaves)):
            if not isinstance(source, torch.Tensor):
                continue
            if self.reuse_unchanged_inputs:
                version = None if source.is_inference() else source._version
                if version is not None and self.sources[index] is source and (self.versions[index] == version):
                    continue
                (self.sources[index], self.versions[index]) = (source, version)
            destination.copy_(source)
        self.graph.replay()
        self.replays += 1
        (outputs, output_spec) = tree_flatten(self.outputs)
        return tree_unflatten(
            [value.clone() if isinstance(value, torch.Tensor) else value for value in outputs], output_spec
        )


class TextEncodeGraph:
    def __init__(self, encoder):
        self.encoder = encoder
        self.guard = None
        self.prompt_key = self.prompt_value = None
        self.hits = self.misses = 0
        self.graph = CudaGraphForward(self._encode)

    def _encode(self, ids, mask):
        context = self.encoder(ids, mask)
        lengths = mask.gt(0).sum(dim=1).long()
        padding = torch.arange(ids.shape[1], device=ids.device)[None, :] >= lengths[:, None]
        return (context.masked_fill(padding[:, :, None], 0), lengths)

    def close(self):
        self.graph.reset()
        self.guard = None
        self.prompt_key = self.prompt_value = None

    def _check_encoder(self):
        if self.guard is None:
            self.guard = module_guard(self.encoder, track_versions=True)
        if not self.guard.all_eval():
            raise ValueError("Prepared text encoding requires all encoder modules in eval mode.")
        if self.guard.changed():
            self.graph.reset()
            self.prompt_key = self.prompt_value = None

    @torch.no_grad()
    def encode_prompt(self, prompt, tokenizer, device):
        from WAMInfer.preparation import encode_text

        self._check_encoder()
        key = (prompt, id(tokenizer), torch.device(device))
        if key != self.prompt_key:
            self.prompt_value = encode_text(prompt, tokenizer=tokenizer, graph=self, device=device)
            self.prompt_key = key
            self.misses += 1
        else:
            self.hits += 1
        return self.prompt_value

    @torch.no_grad()
    def __call__(self, ids, mask):
        self._check_encoder()
        return self.graph(ids, mask)


def _first_frame_conv(module, x, cache_x=None):
    if cache_x is None and x.shape[2] == 1:
        return F.conv2d(
            x[:, :, 0].contiguous(memory_format=torch.channels_last),
            module._inference_2d_weight,
            module.bias,
            module.stride[1:],
            (module._padding[2], module._padding[0]),
            module.dilation[1:],
            module.groups,
        ).unsqueeze(2)
    return CausalConv3d.forward(module, x, cache_x)


class VAEEncodeGraph:
    def __init__(self, vae):
        self.vae = vae
        self.graph = CudaGraphForward(self._encode)
        self.guard = None
        self.convolutions = []

    def close(self):
        self.graph.reset()
        self.guard = None
        for module in self.convolutions:
            if hasattr(module, "_inference_2d_weight"):
                del module._inference_2d_weight
        self.convolutions = []

    def _prepare_convolutions(self):
        self.convolutions = []
        for name, module in self.vae.model.named_modules():
            if (
                isinstance(module, CausalConv3d)
                and (name.startswith("encoder.") or name == "conv1")
                and (module._padding[4] == module.dilation[0] * (module.kernel_size[0] - 1))
                and (module.weight.dtype == torch.bfloat16)
            ):
                module.register_buffer(
                    "_inference_2d_weight",
                    module.weight[:, :, -1].detach().contiguous(memory_format=torch.channels_last),
                    persistent=False,
                )
                self.convolutions.append(module)

    @contextmanager
    def _first_frame_convolutions(self):
        originals = [(module, module.__dict__.get("forward")) for module in self.convolutions]
        try:
            for module, _ in originals:
                module.forward = MethodType(_first_frame_conv, module)
            yield
        finally:
            for module, original in originals:
                if original is None:
                    del module.forward
                else:
                    module.forward = original

    def _encode(self, videos, scale):
        if videos.shape[2] == 1 and videos.dtype == torch.bfloat16:
            with self._first_frame_convolutions():
                return self.vae.model.encode(videos, scale)
        return self.vae.model.encode(videos, scale)

    @torch.no_grad()
    def __call__(self, videos):
        if self.vae.training or videos.device.type != "cuda":
            raise ValueError("VAE encoding Graph requires an eval native CUDA VAE.")
        if self.guard is None or self.guard.changed():
            self.graph.reset()
            self._prepare_convolutions()
            self.guard = module_guard(self.vae.model, track_versions=True)
        scale = self.vae.scale
        if isinstance(scale, (list, tuple)):
            scale = tuple((s.to(device=videos.device) if isinstance(s, torch.Tensor) else s for s in scale))
        elif isinstance(scale, torch.Tensor):
            scale = scale.to(device=videos.device)
        return self.graph(videos, scale)
