# WAMInfer

**OpenWAM inference runtime — initial partial code release for RealtimeWAM.**

[Paper](https://arxiv.org/abs/2610.10079) · [中文源码导读](CODE_GUIDE.md) · [Validation](VALIDATION.md) · [Citation](#citation)

> **RealtimeWAM: How Fast Can I Run My World Action Model?**
>
> Huanan Liu, Ye Li, Kangye Ji, Xiaoyu Chen, Hanyun Cui, Yutian Shen, Yuan Meng, Chenglei Wu, Jingyan Jiang, Bo Li, and Zhi Wang
>
> arXiv:2610.10079, 2026

WAMInfer adds an external Parallel inference path to OpenWAM, sharing the original model and weights. It combines prepared conditioning, request-local clean-frame K/V reuse, compiled phases, fused operators, CUDA Graph replay, and Video/Action stream scheduling. No edits to upstream source files are required.

中文说明：这是论文的首批部分代码发布，当前提供 OpenWAM 的 Parallel 推理路径；完整论文方法中的 FastWAM、跨观测 token 复用及运动自适应计算尚未包含。代码结构与执行原理见[中文导读](CODE_GUIDE.md)。

## Release scope

**v0.1.0 publishes the existing OpenWAM runtime, not a complete reproduction of the paper.**

| Component | This release |
| --- | --- |
| External OpenWAM runtime, original weights and sampler updates | Included |
| Video/Action stream overlap and input-preparation overlap | Included |
| Clean-frame K/V reuse across denoising steps within one request | Included; rebuilt for each observation |
| Triton attention, normalization/RoPE, and prepared FFN execution | Included |
| CUDA Graph replay, cache invalidation, tests, and timing evidence | Included |
| FastWAM backend | Not included |
| Cross-observation selective FFN/token reuse and capacity filling | Not included |
| Motion-adaptive 2F/4F Transformer residual refresh | Not included |
| Full paper benchmark and ablation runners | Not included |

The paper combines dependency-aware scheduling, hardware-aware token reuse, and motion-adaptive refinement. Its 2F/4F settings perform two/four full Transformer evaluations within ten Euler updates. This runtime instead evaluates every Action block at every configured sampler step. Its Video/Action overlap should not be equated with a verified reproduction of the paper's complete Prefill/prediction schedule.

The paper reports **24.09 ms / 8.90×** for FastWAM and **63.09 ms / 10.67×** for OpenWAM, averaged over its benchmark settings. Those are results of the full framework; they are **not performance claims for this release**. See the [paper](https://arxiv.org/abs/2610.10079) for the complete protocol.

## Installation

Start with a working OpenWAM environment and an upstream-compatible checkpoint directory. This package targets upstream revision [`898e2f96`](https://github.com/OpenWAM-Official/OpenWAM/tree/898e2f96c02c172f078c17ff85a055a6bbd419a2); it does not install or bundle OpenWAM itself.

The validated environment uses Python 3.10/3.11, PyTorch 2.7.1+cu128 and its corresponding Triton, plus a C++ compiler, CUDA development tools, and Ninja. The Ada FFN extension also uses the `nvidia.cublas` installation supplied with the PyTorch environment. Install without replacing that environment's dependencies:

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/Na2Ssss/WAMInfer.git
cd WAMInfer
python -m pip install --no-deps --no-build-isolation -e .
```

Alternatively, copy the `WAMInfer/` directory into the upstream OpenWAM root. `import openwam` must already work. Model weights and datasets are supplied separately under their respective terms.

## Usage

```python
from WAMInfer import OpenWAM

model = OpenWAM("/path/to/checkpoint")
result = model.generate(prompt, image, proprio=state)
actions = result["actions"]
model.close()
```

`image` is a PIL image and `state` follows the upstream deployment proprio interface. Defaults are 384×320 pixels, 9 video frames, 32 output actions, 10 sampler steps, and seed 42. Set `decode_video=True` to return video. The original architecture remains available as `model.original`.

To wrap an already loaded native architecture:

```python
from WAMInfer import accelerate

fast = accelerate(architecture)
result = fast.generate(
    schedule, prompt, first_frame_image=image,
    proprio=architecture.normalize_deploy_proprio(state),
    action_num_frames=33,
)
architecture = fast.close()
```

The direct runtime takes an explicit synchronous schedule. The convenience `OpenWAM.generate` interface constructs the native schedule from `num_inference_steps`.

## Execution and numerical contract

- CUDA BF16, `head_dim=128`, Wan22Ti2v, `joint_self_attn` Action, and the native Wan VAE; Parallel is the only inference mode.
- Clean-frame reuse requires `first_frame_causal` attention, one clean observation, and the corresponding zero-timestep conditioning. This cache is rebuilt for every request.
- Calls on one runtime are serialized because Graph buffers and workspaces are reused. Parallelism overlaps branches inside a request.
- Recent incremental changes were checked for byte equality against **the existing Parallel runtime with fixed FFN algorithms**. Historical BF16 fusions differ numerically from native eager; this package does not claim bitwise equality with native inference.
- FFN preparation autotunes its algorithms. Reproducing the exact audited outputs requires the fixed configurations in [verification.json](evidence/verification.json); independent autotuning may select different algorithms.
- Internal `_triton_*` operators rely on their callers for valid shape, layout, and dtype. Runtime checks for cache invalidation and Graph input refresh remain active.

## Measurements for this code

RTX 4090 with 48 GiB memory, BF16, 384×320, 9 video frames, 32 actions, and 10 steps. Timing covers CPU image/state inputs through CPU physical actions, including preprocessing, with a warm text cache. Loading, first compilation/capture, video decoding, and networking are excluded.

| Independent paired experiment | Mean baseline → candidate latency | Pairs |
| --- | ---: | ---: |
| Overlap Action with Video FFN | 221.383 → 217.134 ms | 120 |
| Prepare action-mask indices on CPU | 217.281 → 217.124 ms | 120 |

These are incremental comparisons from separate experiments, not speedups over native eager or reproduction of the paper tables. The latest measured change saved 0.157 ms (0.072%). Removing internal argument checks subsequently reduced source size without a new latency claim.

[Validation](VALIDATION.md) documents numerical boundaries, GPU tests, and measurement conditions. [latency.json](evidence/latency.json) contains the paired samples; [verification.json](evidence/verification.json) records source hashes, FFN configurations, and verification results. Checkpoints, request fixtures, reference action arrays, and raw Nsight traces are not bundled, so the complete historical audits are not self-contained reproductions in this release.

## Tests and benchmark

In the configured OpenWAM environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -c WAMInfer/pytest.ini WAMInfer/tests -q -p no:cacheprovider

python -m WAMInfer.benchmark \
  --checkpoint /path/to/checkpoint \
  --input /path/to/request.npz \
  --prompt-file /path/to/prompt.txt
```

Tests require CUDA and native OpenWAM dependencies. The NPZ contains an HWC uint8 `image` and `proprio`. The benchmark defaults to 10 warmups and 30 measured requests. It measures warm-request latency on your machine; it does not run closed-loop task-success evaluation or replace fixed-algorithm numerical checks.

## Code map

The core contains **9 Python files / 2021 lines** and **2 C++ files / 411 lines**, including comments and blank lines, excluding tests and the benchmark.

| Files | Responsibility |
| --- | --- |
| `runtime.py`, `preparation.py` | Request lifecycle, sampler, inputs, conditioning, clean-frame preparation |
| `blocks.py`, `joint.py` | Model adapters, compiled phases, joint attention, branch scheduling |
| `graphs.py` | CUDA Graph buffers/replay and text/VAE execution |
| `ffn.py`, `_cublaslt_inference.cpp` | Prepared FFN weights, cuBLASLt plans and workspace |
| `_triton_inference.py`, `_triton_attention.py` | Fused GPU operators |
| `_inference_guards.cpp` | CPU-side model and Graph input state checks |

Read the [Chinese source walkthrough](CODE_GUIDE.md) for the full execution flow.

## Citation

If you use this code, please cite the RealtimeWAM paper. Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

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

## License and provenance

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The initial import `846ce85` preserved the Python/C++ source from validated external snapshot `76bade23494774278dbc6e036d5793214adff191`. Commit `c8372cd` removed 75 lines of internal argument checks and unused variables while preserving arithmetic and launch settings. Source hashes are recorded in the verification evidence.

The external-addon organization was informed by [BAC](https://github.com/ky-ji/BAC/tree/82029a6fb0573fd07f4b26088219b0eb5ccc5a67). This package does not implement BAC's approximate block-skipping policy. Upstream model source, weights, and datasets are not distributed here.
