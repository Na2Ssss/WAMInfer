"""Per-model BF16 FFN plans, selected before compilation and graph capture.

Weights stay BF16 and accumulation stays FP32. Bias and tanh GELU share the
up-projection epilogue, avoiding the intermediate BF16 rounding boundary.
Plans/workspaces belong to one serialized model, never a process-wide shape
cache; multiple models cannot overwrite each other's captured workspace.
"""

from functools import lru_cache
from itertools import count
from pathlib import Path

import torch

from WAMInfer.blocks import _parameter_signature

_PLANS = {}
_IDS = count()


@lru_cache(maxsize=1)
def _extension():
    import nvidia.cublas
    from torch.utils.cpp_extension import load

    root = Path(nvidia.cublas.__path__[0])
    return load(
        name="wam_infer_ffn_v1",
        sources=[str(Path(__file__).with_name("_cublaslt_inference.cpp"))],
        extra_include_paths=[str(root / "include")],
        extra_ldflags=[str(root / "lib/libcublasLt.so.12")],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


@torch.library.custom_op("wam_infer::prepared_ffn_linear_v1", mutates_args=())
def prepared_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, plan_id: int) -> torch.Tensor:
    return _PLANS[plan_id].run(x, weight, bias)


@prepared_linear.register_fake
def _fake_linear(x, weight, bias, plan_id):
    return torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)


class PreparedFFN:
    def __init__(self, backbone):
        self.backbone = backbone
        self.weights = {}
        self.signatures = {}
        self.plans = {}
        self.keys = {}

    def _plan(self, weight, bias, rows, gelu):
        key = (weight.device, rows, tuple(weight.shape), weight.stride(), gelu)
        if key not in self.keys:
            if len(self.keys) >= 32:
                raise RuntimeError("FFN preparation supports at most 32 device/shape plans per model.")
            generator = torch.Generator(device=weight.device).manual_seed(3901)
            x = torch.randn(rows, weight.shape[1], device=weight.device, dtype=weight.dtype, generator=generator)
            plan = _extension().Plan(x, weight, bias, True, gelu)
            plan_id = next(_IDS)
            _PLANS[plan_id] = plan
            self.keys[key] = plan_id
        return self.keys[key]

    def prepare(self, rows):
        if torch.is_grad_enabled() or self.backbone.training:
            raise RuntimeError("FFN preparation requires eval mode and disabled gradients.")
        changed = False
        for layer_id, block in enumerate(self.backbone.dit.blocks):
            ffn = block.ffn
            if not isinstance(ffn[1], torch.nn.GELU) or ffn[1].approximate != "tanh":
                raise ValueError("Fused FFN requires tanh GELU.")
            packed = []
            for index in (0, 2):
                layer = ffn[index]
                if layer.weight.dtype != torch.bfloat16 or layer.weight.device.type != "cuda" or layer.bias is None:
                    raise ValueError("Fused FFN requires CUDA BF16 linear layers with bias.")
                key = (layer_id, index)
                signature = _parameter_signature(layer.weight)
                if self.signatures.get(key) != signature:
                    self.weights[key] = layer.weight.detach().T.contiguous().T
                    self.signatures[key] = signature
                    changed = True
                packed.append(self.weights[key])
            for m in rows:
                key = (layer_id, m)
                ids = (self._plan(packed[0], ffn[0].bias, m, True), self._plan(packed[1], ffn[2].bias, m, False))
                changed |= self.plans.get(key) != ids
                self.plans[key] = ids
        return changed

    def forward(self, layer_id, inputs):
        ffn = self.backbone.dit.blocks[layer_id].ffn
        rows = inputs.numel() // inputs.shape[-1]
        if (layer_id, rows) not in self.plans:
            # CFG may merge batches after request preparation. Keep the native
            # FFN for an unprepared geometry instead of tuning during capture.
            return ffn(inputs)
        up, down = self.plans[layer_id, rows]
        x = inputs.reshape(rows, inputs.shape[-1])
        x = prepared_linear(x, self.weights[layer_id, 0], ffn[0].bias, up)
        x = prepared_linear(x, self.weights[layer_id, 2], ffn[2].bias, down)
        return x.reshape(*inputs.shape[:-1], x.shape[-1])

    def __del__(self):
        self.close()

    def close(self):
        for plan_id in self.keys.values():
            _PLANS.pop(plan_id, None)
        self.keys.clear()
        self.plans.clear()
        self.weights.clear()
        self.signatures.clear()
