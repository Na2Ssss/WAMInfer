from types import SimpleNamespace

import pytest
import torch

from WAMInfer.ffn import PreparedFFN, _PLANS


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_ffn_owns_workspace_refreshes_weights_and_releases_plans():
    torch.manual_seed(24)
    ffn = (
        torch.nn.Sequential(torch.nn.Linear(128, 512), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(512, 128))
        .cuda()
        .bfloat16()
        .eval()
    )
    backbone = SimpleNamespace(training=False, dit=SimpleNamespace(blocks=[SimpleNamespace(ffn=ffn)]))
    before = set(_PLANS)
    first, second = PreparedFFN(backbone), PreparedFFN(backbone)
    try:
        first.prepare((16,))
        second.prepare((16,))
        assert set(first.keys.values()).isdisjoint(second.keys.values())
        x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        merged = torch.cat((x, x), dim=0)
        torch.testing.assert_close(first.forward(0, merged), ffn(merged), atol=0, rtol=0)
        expected = ffn(x)
        actual = first.forward(0, x)
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
        torch.testing.assert_close(second.forward(0, x), expected, atol=0.005, rtol=0.03)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                first.forward(0, x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = first.forward(0, x)
        graph.replay()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        graph.reset()
        ffn[0].weight.add_(0.1)
        assert first.prepare((16,))
        updated = first.forward(0, x)
        assert not torch.equal(updated, actual)
        torch.testing.assert_close(updated, ffn(x), atol=0.01, rtol=0.04)
        assert not first.prepare((16,))
        first.close()
        assert set(_PLANS) == before | set(second.keys.values())
    finally:
        torch.cuda.synchronize()
        first.close()
        second.close()
    assert set(_PLANS) == before
