from types import SimpleNamespace

import numpy as np
import pytest
import torch

from WAMInfer import accelerate
from WAMInfer.approximation import MotionRefinement, TokenReuse, select_tokens
from WAMInfer.tests.helpers import make_arch, make_inputs


def test_capacity_and_feature_selection():
    history = TokenReuse((1, 2, 3, 4))
    inputs = dict(latents=torch.zeros(1, 4, 3, 4, 4))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    reuse, observation = history.begin(image, inputs, "task", (1, 2, 2))
    assert reuse["count"] == 4
    previous = torch.ones(1, 4, 8)
    history.commit(observation, previous, (previous,))
    image[:4, 4:] = 30  # Token 1 is mandatory, regardless of its feature score.
    reuse, _ = history.begin(image, inputs, "task", (1, 2, 2))
    assert reuse["count"] == 2
    state = SimpleNamespace(extras=dict(token_reuse=reuse))
    current = torch.ones(1, 12, 8)
    current[:, 2:4] = 5  # Equal scores: smaller spatial index wins.
    select_tokens(state, current)
    assert state.extras["token_indices"].tolist() == [1, 2, *range(4, 12)]
    torch.testing.assert_close(history.features, previous)
    assert history.begin(image, inputs, "new task", (1, 2, 2))[0]["count"] == 4


def test_motion_schedule_and_contract():
    schedule = [(1000 - i * 100,) * 2 for i in range(11)]
    for motion, expected in ((0.081, [0, 3]), (0.08, [0, 3, 6, 9])):
        gate = MotionRefinement(lambda actions: motion, 0.08, schedule)
        before = [i for i in range(4) if gate.full(i)]
        gate.decide(np.zeros((32, 7)))
        assert before + [i for i in range(4, 10) if gate.full(i)] == expected
    with pytest.raises(ValueError, match="motion_metric"):
        MotionRefinement(None, 0.08, schedule)
    with pytest.raises(ValueError, match="10 Euler"):
        MotionRefinement(lambda x: 0.1, 0.08, schedule[:5])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_reuse_executes_compact_ffn_with_current_gates():
    fast = accelerate(make_arch("cuda", torch.bfloat16, head_dim=128))
    vb = fast.video_backbone
    vb.ffn = None  # Inspect actual linear row counts independently of the prepared backend.
    n, count, length, dim = 4, 2, 12, vb.dit.blocks[1].ffn[0].in_features
    x = torch.randn(1, length, dim, device="cuda", dtype=torch.bfloat16)
    normalized = torch.randn_like(x)
    old = torch.randn(1, n, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([0, 3, *range(n, length)], device="cuda")
    state = SimpleNamespace(
        time_mod=torch.randn(1, length, 6, dim, device="cuda", dtype=x.dtype),
        extras=dict(token_reuse=dict(n=n, count=count, cache=(old,)), token_indices=indices, token_outputs=[]),
    )
    rows = []
    hook = vb.dit.blocks[1].ffn[0].register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    vb.ffn_at_layer_for_compile(1, state, normalized, x)
    hook.remove()
    assert rows == [length - n + count]
    clean = state.extras["token_outputs"][0]
    torch.testing.assert_close(clean[:, 1:3], old[:, 1:3], atol=0, rtol=0)
    output = vb.dit.blocks[1].ffn(normalized)
    output[:, :n] = clean
    gate = (vb.dit.blocks[1].modulation + state.time_mod)[:, :, 5]
    torch.testing.assert_close(state.hidden_states, x + gate * output, atol=0.02, rtol=0.02)
    fast.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reuse_tokens,adaptive", [(False, True), (True, False), (True, True)])
@torch.no_grad()
def test_switches_graph_replay_and_history(monkeypatch, reuse_tokens, adaptive):
    from torch.utils._pytree import tree_unflatten

    fast = accelerate(
        make_arch("cuda", torch.bfloat16, head_dim=128),
        reuse_tokens=reuse_tokens,
        adaptive_2f4f=adaptive,
        refresh_buckets=(1, 2, 3, 4),
    )
    inputs = make_inputs(fast)
    monkeypatch.setattr(
        fast.video_backbone,
        "preprocess_input_for_inference",
        lambda **kw: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
    )
    monkeypatch.setattr(fast.video_backbone._native, "decode_video", lambda x, **kw: x.clone())
    schedule = [(1000.0 - i * 100,) * 2 for i in range(11)]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    calls, coarse_actions = [], []
    flow_step = fast.action_scheduler.flow_step

    def flow(*args):
        calls.append(1)
        return flow_step(*args)

    monkeypatch.setattr(fast.action_scheduler, "flow_step", flow)

    def run(motion, prompt="task", decode=False):
        calls.clear()

        def metric(actions):
            coarse_actions.append(actions.copy())
            return motion

        result = fast.generate(
            schedule,
            prompt,
            first_frame_image=image,
            proprio=inputs["proprio"],
            action_num_frames=4,
            motion_metric=metric,
            decode_video=decode,
        )
        assert len(calls) == 10
        assert np.isfinite(result["actions"]).all()
        expected = (0, 3) if motion > 0.08 else (0, 3, 6, 9)
        assert fast.last_stats["full_evaluations"] == (expected if adaptive else tuple(range(10)))
        return result

    initial = run(0.1)
    if reuse_tokens:
        assert fast.last_stats["refreshed_tokens"] == 4
    run(0.01)
    if reuse_tokens:
        assert fast.last_stats["refreshed_tokens"] == 1
        captures = fast._first_frame_preparation.variants[1].captures
    replay = run(0.01, decode=True)
    assert torch.isfinite(replay["video"]).all()
    if reuse_tokens:
        assert fast._first_frame_preparation.variants[1].captures == captures
        # Each new observation updates the L0 reference, including graph replay.
        previous = fast.token_reuse.features.clone()
        inputs["first_frame_latents"].add_(0.5)
        run(0.1)
        assert not torch.equal(fast.token_reuse.features, previous)
        inputs["first_frame_latents"].sub_(0.5)
        run(0.1, prompt="new task")
        assert fast.last_stats["refreshed_tokens"] == 4
        fast.reset()
        assert fast.token_reuse.cache is None
        run(0.1)
        assert fast.last_stats["refreshed_tokens"] == 4
        fast.video_backbone.dit.blocks[0].self_attn.q.weight.add_(0.01)
        run(0.1)
        assert fast.last_stats["refreshed_tokens"] == 4
    else:
        np.testing.assert_array_equal(run(0.1)["actions"], initial["actions"])
    if adaptive:
        # 2F and 4F share the very same prefix and coarse prediction.
        if reuse_tokens:
            fast.reset()
            run(0.1)
            fast.reset()
            run(0.01)
            np.testing.assert_array_equal(coarse_actions[-2], coarse_actions[-1])
        else:
            np.testing.assert_array_equal(coarse_actions[0], coarse_actions[1])
        graph = fast._residual_graphs[False]
        args, kwargs = tree_unflatten(graph.inputs, graph.signature[0])
        expected = fast._forward_impl(*args, **kwargs)

        def forbidden(*args):
            raise AssertionError("Cached residual steps must not enter the Transformer")

        monkeypatch.setattr(fast, "_compiled_mot_run_joint_loop", forbidden)
        actual = fast._forward_impl(*args, **kwargs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        # Heads depend on the CURRENT latent; cached velocities would fail this.
        changed = fast._forward_impl(args[0] + 1, **kwargs)
        assert not torch.equal(changed[1], actual[1])
    fast.close()
