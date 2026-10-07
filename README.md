# WAMInfer

OpenWAM 的外接 Parallel 推理加速包。共享原模型和权重，保留原始采样步骤，通过条件预计算、当前帧 K/V 复用、CUDA Graph、分支调度和融合算子降低推理延时。

**核心为 9 个 Python 文件、2096 行，另有 2 个 C++ 扩展、411 行。** 行数包含注释和空行，不包括 benchmark、测试和文档。原生 OpenWAM 源码不在本仓库内。

从 [中文源码导读](CODE_GUIDE.md) 了解一次推理如何经过这些文件；[验证记录](VALIDATION.md) 说明测试条件、数值边界和测量结果。

## 安装与接入

先准备能正常加载 checkpoint 的 OpenWAM 环境。本包对照的原库版本为 [`898e2f96`](https://github.com/OpenWAM-Official/OpenWAM/tree/898e2f96c02c172f078c17ff85a055a6bbd419a2)。已验证环境使用 Python 3.10/3.11、PyTorch 2.7.1+cu128、对应 Triton；还需 C++/CUDA 编译工具及 Ninja，Ada 的 FFN 使用 PyTorch 环境中的 `nvidia.cublas`。

在该环境中安装本仓库；`--no-deps` 保留现有 OpenWAM 依赖版本：

```bash
git clone https://github.com/Na2Ssss/WAMInfer.git
cd WAMInfer
python -m pip install --no-deps --no-build-isolation -e .
```

也可以将仓库中的 `WAMInfer/` 文件夹直接放入 OpenWAM 根目录使用。原库需要已经能够 `import openwam`；本包不自动安装另一份 OpenWAM。

```python
from WAMInfer import OpenWAM

model = OpenWAM("/path/to/checkpoint")
result = model.generate(prompt, image, proprio=state)
actions = result["actions"]
model.close()
```

`image` 为 PIL 图像，`state` 为原部署接口的 proprio。默认 384×320、9 视频帧、32 个输出动作、10 步、seed=42；`decode_video=True` 返回视频。帧数、尺寸和步数可通过 `generate` 参数设置。第一次运行包含加载、编译和 Graph 捕获，不能当作热请求延时。

已有原生 architecture 时：

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

直接 runtime 使用显式 schedule；便捷入口 `OpenWAM.generate` 从 `num_inference_steps` 构造原生同步 schedule。原模型可通过 `model.original` 访问。

## 执行范围和数值边界

- CUDA BF16、head_dim=128、Wan22Ti2v、`joint_self_attn` Action 分支和原生 Wan VAE；只提供 Parallel 路径。
- 当前帧缓存要求 `first_frame_causal`、单张 clean 观测和相应零时间步条件。缓存每请求重建，未来帧和 Action 的必要计算继续执行。
- 同一 runtime 内请求串行化；这是单请求中 Video/Action 的 GPU 并行，不是多请求并发服务。
- 最近的优化按**相对已有 Parallel、固定 FFN 算法的逐字节一致性**验证。历史 BF16 融合相对原生 eager 存在数值差异，不能将整个外接包宣称为与 eager 逐位相同。
- FFN 会在准备阶段选择算法；完整性能对照固定了算法配置。新进程自动调优不等于自动复现同一组算法，配置见 [验证证据](evidence/verification.json)。

## 已测结果

RTX 4090（48 GiB 配置），BF16，384×320，9 视频帧、32 动作、10 步。CPU 输入到 CPU 动作，含预处理，文本缓存命中；不含加载、首次编译、视频解码和网络传输。

| 独立配对实验 | 基线 → 候选平均延时 | 配对数 |
| --- | ---: | ---: |
| Action 改为与 Video FFN 重叠 | 221.383 → 217.134 ms | 120 |
| 最新 CPU 掩码准备 | 217.281 → 217.124 ms | 120 |

两行来自不同实验，不能相加或跨实验相减。最新改动只省 0.157 ms（0.072%）。这些是增量对照，不能当作相对原生 eager 的总加速比。原始配对样本见 [latency.json](evidence/latency.json)。

## 测试与测速

```bash
python -m WAMInfer.benchmark \
  --checkpoint /path/to/checkpoint \
  --input /path/to/request.npz \
  --prompt-file /path/to/prompt.txt

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -c WAMInfer/pytest.ini WAMInfer/tests -q -p no:cacheprovider
```

NPZ 包含 HWC uint8 `image` 和 `proprio`。默认测速预热 10 次、运行 30 次；它测当前机器的热请求延时，不替代固定算法的数值审计。测试需要 CUDA 和原生 OpenWAM 依赖。

## 来源

计算代码来自已验证外接版本 `76bade23494774278dbc6e036d5793214adff191`，迁入本仓库时保持 Python/C++ 源文件字节不变，哈希见 [verification.json](evidence/verification.json)。

外接组织方式参考 [BAC](https://github.com/ky-ji/BAC/tree/82029a6fb0573fd07f4b26088219b0eb5ccc5a67)，未加入其近似 block 跳算策略。许可证和来源说明见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。
