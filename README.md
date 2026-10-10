<div align="center">

# WAMInfer

**Parallel OpenWAM inference, with two optional acceleration switches.**

[![Paper](https://img.shields.io/badge/arXiv-2610.10079-b31b1b.svg)](https://arxiv.org/abs/2610.10079)
[![Release](https://img.shields.io/badge/release-v0.2.0-2563eb.svg)](https://github.com/Na2Ssss/WAMInfer/releases/tag/v0.2.0)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Quick start](#quick-start) · [Acceleration switches](#acceleration-switches) · [Performance](#performance) · [中文源码导读](CODE_GUIDE.md)

**[RealtimeWAM: How Fast Can I Run My World Action Model?](https://arxiv.org/abs/2610.10079)**<br>
Huanan Liu, Ye Li, Kangye Ji, Xiaoyu Chen, Hanyun Cui, Yutian Shen, Yuan Meng, Chenglei Wu, Jingyan Jiang, Bo Li, and Zhi Wang

</div>

WAMInfer is the OpenWAM inference implementation for **RealtimeWAM**. It wraps the original model, shares its weights, and accelerates inference through compiled phases, fused operators, CUDA Graphs, and concurrent Video/Action execution. **No upstream source edits or additional training are required.**

Parallel is always enabled. Two independent switches control the approximate methods:

| Component | What it does | Default |
| --- | --- | --- |
| **Parallel execution** | Overlaps Video/Action branches and input preparation; reuses observation K/V within a request | Always on |
| **Token reuse** | Refreshes selected observation-token FFNs and reuses the other rows across observations | `reuse_tokens=False` |
| **Adaptive 2F/4F** | Chooses two or four full Transformer evaluations within ten Euler updates | `adaptive_2f4f=False` |

Both switches are opt-in approximations. With both off, the existing Parallel behavior is preserved. This repository currently covers OpenWAM; FastWAM and the complete paper evaluation suite are not included.

中文：默认使用 Parallel；跨观测 token 复用与自适应 2F/4F 分别通过开关启用。代码保持外接形式，原理与文件说明见[中文导读](CODE_GUIDE.md)。

## Installation

Start with a working [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM/tree/898e2f96c02c172f078c17ff85a055a6bbd419a2) environment and a compatible checkpoint. `import openwam` must already work; model weights and datasets are supplied separately.

Validated with **Python 3.10/3.11, PyTorch 2.7.1+cu128 and its matching Triton**, on A100 and RTX 4090. Building the extensions requires a C++ compiler, CUDA development tools, Ninja, and the `nvidia.cublas` package supplied with the PyTorch environment.

```bash
git clone https://github.com/Na2Ssss/WAMInfer.git
cd WAMInfer
python -m pip install --no-deps --no-build-isolation -e .
```

For the tagged release, run `git checkout v0.2.0` before installation. Alternatively, copy the `WAMInfer/` package into your OpenWAM root.

<details>
<summary>Supported model configuration</summary>

- Upstream reference revision: [`898e2f96`](https://github.com/OpenWAM-Official/OpenWAM/tree/898e2f96c02c172f078c17ff85a055a6bbd419a2).
- CUDA BF16, `head_dim=128`, Wan22Ti2v, `joint_self_attn` Action backbone, and the native Wan VAE.
- Request-local observation K/V reuse uses `first_frame_causal` attention and one clean observation with zero-timestep conditioning.
- Calls on one runtime are serialized; parallelism overlaps branches inside each request.

</details>

## Quick start

Use an RGB image prepared for your checkpoint and **raw proprioception in the checkpoint's expected coordinate order**. The convenience API normalizes the state and returns denormalized NumPy actions.

```python
import numpy as np
from PIL import Image
from WAMInfer import OpenWAM

prompt = "Pick up the red block."
image = Image.open("/path/to/observation.png").convert("RGB")  # 320 W × 384 H
state = np.load("/path/to/proprio.npy")

model = OpenWAM("/path/to/checkpoint")
result = model.generate(prompt, image, proprio=state)
actions = result["actions"]
model.close()
```

Defaults: **384×320 image resolution, 9 video frames, 32 output actions, 10 Euler updates, seed 42**. Set `decode_video=True` to also return video. The original model remains available as `model.original`.

Keep one model instance alive across observations to reuse warmed graphs and caches. The first call includes compilation, FFN preparation and graph capture; measure latency after warmup.

<details>
<summary>Wrap an already loaded OpenWAM architecture</summary>

Given a loaded `architecture` and its native synchronous `schedule`:

```python
from WAMInfer import accelerate

fast = accelerate(architecture)
result = fast.generate(
    schedule, prompt,
    first_frame_image=image,
    proprio=architecture.normalize_deploy_proprio(state),
    action_num_frames=33,
)
architecture = fast.close()
```

`accelerate` accepts the same two switches. Unlike `OpenWAM.generate`, this interface takes an explicit schedule and an already normalized state.

</details>

## Acceleration switches

### Token reuse

```python
model = OpenWAM("/path/to/checkpoint", reuse_tokens=True)
result = model.generate(prompt, image, proprio=state)
```

The first observation fills the cache. Subsequent observations compute layer 0 in full, then refresh selected current-observation FFN rows in later layers. Attention, current residual gates, future-video tokens and Action tokens still receive fresh computation at each full evaluation.

Call **`model.reset()` when starting a new episode or changing the camera/preprocessing configuration**. Prompt, weight and input-geometry changes invalidate history automatically. Reuse requires one RGB observation and future video frames, as in the default request.

<details>
<summary>How refresh positions are selected</summary>

1. Mean absolute RGB change above 5/255 marks a position for mandatory refresh.
2. Reserve 25% of the remaining positions, then round the total up to an execution capacity.
3. Fill the remaining capacity with positions whose layer-0 features changed most; ties follow spatial index order.

The default is `refresh_buckets=(32, 64, 96, 120)` for 120 observation tokens. Other geometries retain smaller configured capacities and always include full refresh. Capacities should be profiled for your GPU and execution shapes. Cache history is committed once per completed request; graph warmup does not advance it.

</details>

### Adaptive 2F/4F

**2F/4F counts full Transformer evaluations, not sampler updates.** Both paths retain ten Euler updates and compute the current output heads at every update.

| Predicted displacement | Full Transformer evaluations, zero-indexed |
| --- | --- |
| Above `motion_threshold=0.08` metres | 0, 3 — **2F** |
| At or below the threshold | 0, 3, 6, 9 — **4F** |

Both paths share the first two evaluations. After evaluation 3, the runtime decodes a coarse action prediction and calls `motion_metric(actions)` once on CPU. Other updates embed current latents and add cached Transformer residuals before running the output heads.

The metric must match your robot's action representation. For **absolute end-effector XYZ in metres in columns 0:3 of both actions and state**, save this example as `motion_metric.py`:

```python
import numpy as np


def displacement(actions, raw_proprio):
    predicted_xyz = actions[:32, :3]
    current_xyz = np.asarray(raw_proprio)[..., :3]
    return float(np.linalg.norm(predicted_xyz - current_xyz, axis=-1).max())
```

Then enable either adaptive refinement alone or both switches:

```python
from motion_metric import displacement

model = OpenWAM(
    "/path/to/checkpoint",
    reuse_tokens=True,       # Independent; set False for adaptive refinement alone.
    adaptive_2f4f=True,
)
result = model.generate(
    prompt, image, proprio=state,
    motion_metric=lambda actions: displacement(actions, state),
)
print(model.architecture.last_stats)
model.close()
```

Adaptive refinement requires `num_inference_steps=10`. For two arms, take the maximum displacement over both arms. Joint actions require forward kinematics; delta actions require the controller's clipping, scaling and cumulative displacement. Use the XYZ example only for the stated action representation. `last_stats` reports full-evaluation indices, refreshed token count and displacement in metres.

## Performance

**RTX 4090 with 48 GiB memory · BF16 · 384×320 · 9 video frames · 32 actions · 10 Euler updates.**

Each setting below repeats one observation for 30 warm requests using fixed FFN algorithms. Timing covers **CPU image/state inputs → denormalized CPU actions**, including preprocessing. Model loading, compilation/capture, video decoding and networking are excluded; the text cache is warm.

| Token reuse | Adaptive schedule | Mean latency |
| :---: | --- | ---: |
| Off | Off — default Parallel, 10 full evaluations | **211.76 ms** |
| On | Off — 10 full evaluations | 211.45 ms |
| Off | Forced 2F | 57.46 ms |
| Off | Forced 4F | 96.25 ms |
| On | Forced 2F | 58.53 ms |
| On | Forced 4F | 96.33 ms |

The gate was forced to return 0.10 m or 0.01 m to exercise both schedules. Token reuse refreshed 32/120 observation positions after initialization. **This single-observation experiment shows no clear additional latency benefit from token reuse** in the current batched Parallel path; the large reduction comes from fewer Transformer evaluations.

These are offline execution measurements. Closed-loop task success has not been rerun for this release, and these timings do not reproduce the paper's benchmark averages. See [raw samples and configurations](evidence/switches.json) and [validation, numerical comparisons and earlier profiling](VALIDATION.md).

## Benchmark and validation

In the configured OpenWAM environment:

```bash
python -m WAMInfer.benchmark \
  --checkpoint /path/to/checkpoint \
  --input /path/to/request.npz \
  --prompt-file /path/to/prompt.txt
```

The NPZ contains `image` (HWC uint8 RGB) and `proprio` (raw state). Defaults are 10 warmups and 30 measured requests. Add `--reuse-tokens` independently, or `--adaptive-2f4f --motion-metric motion_metric:displacement` using a robot-appropriate metric such as the qualified example above. The CLI callback takes **both** `(actions, raw_proprio)`; the Python `generate` callback takes `actions` only. This benchmark repeats one input, so it does not measure changing-scene or closed-loop behavior.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -c WAMInfer/pytest.ini WAMInfer/tests -q -p no:cacheprovider
```

**20 tests passed on each of A100 and RTX 4090.** With both switches disabled, 12 regression fixture outputs matched v0.1.0 byte for byte; the complete checkpoint also matched its prior fixed-FFN Parallel reference. The optional methods deliberately introduce approximation. Historical BF16 fusions already differ from native eager, so byte equality with the original native model is not claimed. FFN autotuning can choose different accumulation orders; audited configurations are recorded with the [evidence](evidence/switches.json).

## Code map

**10 core Python files / 2,231 lines**, plus **2 C++ files / 411 lines**; counts include comments and blank lines, excluding tests and the benchmark.

| Entry points | Responsibility |
| --- | --- |
| [runtime.py](WAMInfer/runtime.py), [preparation.py](WAMInfer/preparation.py) | Public API, sampler, input preparation and request-local caches |
| [approximation.py](WAMInfer/approximation.py) | Cross-observation token selection and motion-budget decisions |
| [blocks.py](WAMInfer/blocks.py), [joint.py](WAMInfer/joint.py) | Model adapters, compiled phases and Video/Action stream scheduling |
| [graphs.py](WAMInfer/graphs.py) | CUDA Graph replay and text/VAE execution |
| [ffn.py](WAMInfer/ffn.py), [_cublaslt_inference.cpp](WAMInfer/_cublaslt_inference.cpp) | Prepared FFN weights, GEMM algorithms and workspaces |
| [_triton_inference.py](WAMInfer/_triton_inference.py), [_triton_attention.py](WAMInfer/_triton_attention.py) | Fused normalization, RoPE and attention kernels |
| [_inference_guards.cpp](WAMInfer/_inference_guards.cpp) | Model and graph-input change detection |

Start with the [中文源码导读](CODE_GUIDE.md) for a walkthrough of one complete inference request.

## Citation

If you use WAMInfer, please cite RealtimeWAM. Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{liu2026realtimewam,
  title         = {RealtimeWAM: How Fast Can I Run My World Action Model?},
  author        = {Huanan Liu and Ye Li and Kangye Ji and Xiaoyu Chen and Hanyun Cui and Yutian Shen and Yuan Meng and Chenglei Wu and Jingyan Jiang and Bo Li and Zhi Wang},
  year          = {2026},
  eprint        = {2610.10079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.10079},
  url           = {https://arxiv.org/abs/2610.10079}
}
```

## License and acknowledgements

[Apache-2.0](LICENSE). WAMInfer builds on [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM); the external-addon organization was informed by [BAC](https://github.com/ky-ji/BAC/tree/82029a6fb0573fd07f4b26088219b0eb5ccc5a67). See [NOTICE](NOTICE) for provenance. Upstream weights and datasets remain subject to their respective terms.
