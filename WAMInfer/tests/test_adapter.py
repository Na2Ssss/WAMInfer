import numpy as np
import pytest
import torch

from WAMInfer import accelerate
from WAMInfer.tests.helpers import make_arch, make_inputs


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_external_runtime_refresh_and_native_restoration(monkeypatch):
    native = make_arch("cuda", torch.bfloat16, head_dim=128)
    inputs = make_inputs(native)
    original = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def prepare(**kwargs):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    monkeypatch.setattr(native.video_backbone, "preprocess_input_for_inference", prepare)
    schedule = [(float(1000 - 100 * i), float(1000 - 100 * i)) for i in range(11)]

    def run(model):
        return model.generate(
            schedule,
            "test",
            proprio=inputs["proprio"],
            action_num_frames=4,
            **({"num_inference_steps": len(schedule) - 1} if model is native else {}),
            decode_video=False,
        )["actions"]

    reference = run(native)
    parameters = {k: p.data_ptr() for k, p in native.named_parameters()}
    keys = set(native.state_dict())
    methods = [(module, module.forward.__func__) for module in native.modules()]
    generate = native.generate.__func__
    fast = accelerate(native)
    monkeypatch.setattr(fast.video_backbone, "preprocess_input_for_inference", prepare)
    first = run(fast)
    np.testing.assert_allclose(first, reference, atol=0.03, rtol=0.03)
    for name in ("first_frame_latents", "context", "proprio"):
        inputs[name].add_(0.25)
    assert not np.array_equal(run(fast), first)
    inputs.update(original)
    np.testing.assert_array_equal(run(fast), first)
    for k, v in fast._first_frame_preparation.graph.outputs[2][0]:
        k.fill_(float("nan"))
        v.fill_(float("nan"))
    np.testing.assert_array_equal(run(fast), first)
    assert {k: p.data_ptr() for k, p in native.named_parameters()} == parameters
    assert native.generate.__func__ is generate
    assert all(module.forward.__func__ is method for module, method in methods)
    assert set(native.state_dict()) == keys
    fast.close()
    np.testing.assert_array_equal(run(native), reference)
    assert not any("_inference_" in name for name, _ in native.named_buffers())
    with pytest.raises(RuntimeError, match="closed"):
        run(fast)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_external_weight_mutation_and_schedule_refresh(monkeypatch):
    native = make_arch("cuda", torch.bfloat16, head_dim=128)
    inputs = make_inputs(native)
    fast = accelerate(native)
    monkeypatch.setattr(
        fast.video_backbone,
        "preprocess_input_for_inference",
        lambda **kw: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
    )
    schedule = [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)]

    def run():
        return fast.generate(schedule, "test", proprio=inputs["proprio"], action_num_frames=4)["actions"]

    first = run()
    captures = fast._action_cuda_graph_forward.captures
    native.video_backbone.dit.blocks[0].self_attn.q.weight.add_(0.03)
    native.action_backbone.blocks[0].cross_attn.k.weight = torch.nn.Parameter(
        native.action_backbone.blocks[0].cross_attn.k.weight.detach() + 0.02
    )
    assert not np.array_equal(run(), first)
    assert fast._action_cuda_graph_forward.captures == captures + 1
    assert fast._conditioning_preparation.time_builds == 2
    schedule[:] = [(1000.0, 900.0), (800.0, 600.0), (0.0, 0.0)]
    assert np.isfinite(run()).all()
    assert fast._conditioning_preparation.time_builds == 3
    fast.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frames", [1, 3])
@torch.no_grad()
def test_final_action_graph_omits_only_dead_video_tail(monkeypatch, frames):
    from torch.utils._pytree import tree_unflatten

    fast = accelerate(make_arch("cuda", torch.bfloat16, head_dim=128))
    inputs = make_inputs(fast)
    inputs["latents"] = inputs["latents"][:, :, :frames].contiguous()
    monkeypatch.setattr(
        fast.video_backbone,
        "preprocess_input_for_inference",
        lambda **kw: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
    )
    monkeypatch.setattr(fast.video_backbone._native, "decode_video", lambda x, **kw: x.clone())
    for schedule in ([(1000.0, 1000.0), (0.0, 0.0)], [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)]):
        kwargs = dict(proprio=inputs["proprio"], action_num_frames=4)
        actions = fast.generate(schedule, "test", **kwargs)["actions"]
        decoded = fast.generate(schedule, "test", decode_video=True, **kwargs)
        np.testing.assert_array_equal(actions, decoded["actions"])
        assert torch.isfinite(decoded["video"]).all()
    graph = fast._action_cuda_graph_forward
    args, kwargs = tree_unflatten(graph.inputs, graph.signature[0])
    expected = fast._forward_impl(*args, **dict(kwargs, _inference_action_only=False))[1]

    def unused(*args, **kwargs):
        raise AssertionError("Final action prediction must not execute the unused Video tail")

    loop = fast._compiled_mot_run_joint_loop
    video_post = loop.video_post

    def checked_video_post(layer, *args):
        assert layer != loop.driver.num_layers - 1, "Final action output must skip Video projections too"
        return video_post(layer, *args)

    monkeypatch.setattr(loop, "video_post", checked_video_post)
    monkeypatch.setattr(loop, "video", loop.video[:-1] + [unused])
    monkeypatch.setattr(fast.video_backbone, "finalize", unused)
    graph.reset()
    video, actual = graph(*args, **kwargs)
    assert video is None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    fast.close()
