import pytest
import torch

from WAMInfer._triton_attention import masked_attention
from WAMInfer.graphs import CudaGraphForward


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_unchanged_graph_inputs_refresh_after_mutation_and_keep_output_ownership():
    graph = CudaGraphForward(lambda x, context: x + context, reuse_unchanged_inputs=True)
    x = torch.ones(4, device="cuda")
    context = torch.full_like(x, 2)
    retained = []
    for tensor in (None, x, context, None):
        if tensor is not None:
            tensor.add_(1)
        result = graph(x, context)
        torch.testing.assert_close(result, x + context, atol=0, rtol=0)
        retained.append((result, result.clone()))
    with torch.inference_mode():
        # Tensors without version counters must always be copied.
        fresh = torch.zeros_like(x)
        for _ in range(2):
            fresh.add_(1)
            torch.testing.assert_close(graph(fresh, context), fresh + context, atol=0, rtol=0)
    for result, snapshot in retained:
        torch.testing.assert_close(result, snapshot, atol=0, rtol=0)
    assert graph.captures == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_sequential_graphs_share_inputs_and_refresh_after_recapture():
    first = CudaGraphForward(lambda x, context: x + context, reuse_unchanged_inputs=True)
    final = CudaGraphForward(lambda x, context: x * context, shared_inputs=first)
    retained = []
    for size in (4, 8, 4):
        x = torch.ones(size, device="cuda")
        context = torch.full_like(x, 2)
        for reset in (False, True):
            if reset:
                first.reset()
            a = first(x, context)
            retained.append((a, a.clone()))
            context.add_(1)
            b = final(x, context)
            retained.append((b, b.clone()))
            assert first.inputs is final.inputs
            assert first.sources is final.sources
            torch.testing.assert_close(b, x * context, atol=0, rtol=0)
            torch.testing.assert_close(first(x, context), x + context, atol=0, rtol=0)
    for value, snapshot in retained:
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("first,future,action,heads,batch", [(3, 6, 5, 2, 2), (120, 240, 32, 24, 1)])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_current_frame_banks_match_native_attention_and_refresh(first, future, action, heads, batch):
    from WAMInfer._triton_attention import current_frame_attention
    from WAMInfer.tests.helpers import make_arch

    def random(rows):
        return torch.randn(batch, rows, heads * 128, device="cuda", dtype=torch.bfloat16)

    qv, qa, ck, kv, ka, cv, vv, va = [random(n) for n in (future, action, first, future, action, first, future, action)]
    graph = CudaGraphForward(lambda *tensors: current_frame_attention(*tensors, heads))
    mask = make_arch().mot_driver._build_attention_mask(
        s_video=first + future, s_action=action, video_tokens_per_frame=first, device=qv.device
    )[first:]
    before = None
    for change in (0.0, 0.5):
        ck.add_(change)
        cv.sub_(change)
        q = torch.cat((qv, qa), 1)
        k, v = torch.cat((ck, kv, ka), 1), torch.cat((cv, vv, va), 1)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.reshape(batch, future + action, heads, 128).transpose(1, 2),
                k.reshape(batch, first + future + action, heads, 128).transpose(1, 2),
                v.reshape(batch, first + future + action, heads, 128).transpose(1, 2),
                attn_mask=mask,
            )
            .transpose(1, 2)
            .reshape_as(q)
        )
        video, actions = graph(qv, qa, ck, kv, ka, cv, vv, va)
        actual = torch.cat((video, actions), 1)
        torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.02)
        if before is not None:
            assert not torch.equal(actual, before)
        before = actual
    assert graph.captures == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_exact_masked_attention_prefix_holes_and_fresh_graph_inputs():
    torch.manual_seed(42)
    q = torch.randn(1, 32, 3072, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 513, 3072, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 1, 1, 513, device="cuda", dtype=torch.bool)
    graph = CudaGraphForward(lambda q, k, v, mask: masked_attention(q, k, v, mask, 24))
    for length in (0, 1, 24, 128, 512):
        mask.zero_()
        mask[..., :length] = True
        mask[..., 512] = True
        if length == 128:
            mask[..., 9] = False
        actual = graph(q, k, v, mask)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.view(1, 32, 24, 128).transpose(1, 2),
                k.view(1, 513, 24, 128).transpose(1, 2),
                v.view(1, 513, 24, 128).transpose(1, 2),
                attn_mask=mask,
            )
            .transpose(1, 2)
            .reshape_as(q)
        )
        torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.02)
    mask.zero_()
    assert torch.count_nonzero(graph(q, k, v, mask)) == 0
    assert graph.captures == 1


@pytest.mark.parametrize("video,first,action,heads,batch", [(9, 3, 5, 2, 2), (360, 120, 32, 24, 1)])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_structured_joint_attention_matches_full_mask(video, first, action, heads, batch):
    from WAMInfer._triton_attention import joint_first_frame_attention
    from WAMInfer.tests.helpers import make_arch

    count = video + action
    q = torch.randn(batch, count, heads * 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    arch = make_arch(mask="mutual")
    mask = arch.mot_driver._build_attention_mask(
        s_video=video, s_action=action, video_tokens_per_frame=first, device=q.device
    )
    assert not mask[:first, first:].any()
    graph = CudaGraphForward(lambda q, k, v: joint_first_frame_attention(q, k, v, video, first, heads))
    retained = []
    for change in (0.0, 0.1):
        q.add_(change)
        k.add_(change)
        v.sub_(change)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.reshape(batch, count, heads, 128).transpose(1, 2),
                k.reshape(batch, count, heads, 128).transpose(1, 2),
                v.reshape(batch, count, heads, 128).transpose(1, 2),
                attn_mask=mask,
            )
            .transpose(1, 2)
            .reshape_as(q)
        )
        actual = graph(q, k, v)
        torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.02)
        retained.append((actual, actual.clone()))
    for actual, copy in retained:
        torch.testing.assert_close(actual, copy, atol=0, rtol=0)
    # Changing Action keys/values cannot affect clean-frame queries, but must
    # affect the other queries that see them under the native mutual mask.
    before = graph(q, k, v)
    k[:, video:].add_(2)
    v[:, video:].add_(2)
    after = graph(q, k, v)
    torch.testing.assert_close(after[:, :first], before[:, :first], atol=0, rtol=0)
    assert not torch.equal(after[:, first:], before[:, first:])
    assert graph.captures == 1
