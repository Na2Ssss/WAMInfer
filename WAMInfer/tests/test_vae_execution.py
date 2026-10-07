import pytest
import torch

from WAMInfer.graphs import VAEEncodeGraph
from openwam.model.video_backbone.wan.models.vae import VideoVAE38_


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_single_frame_vae_graph_refreshes_pixels_weights_and_scale():
    torch.manual_seed(7)
    vae = torch.nn.Module()
    vae.model = VideoVAE38_(dim=8, dec_dim=8, z_dim=4, num_res_blocks=1).cuda().bfloat16()
    vae.scale = [torch.zeros(4), torch.ones(4)]
    vae.eval()
    runner = VAEEncodeGraph(vae)
    pixels = torch.randn(1, 3, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
    expected = vae.model.encode(pixels, vae.scale)
    methods = [(module, module.forward.__func__) for module in vae.modules()]
    first = runner(pixels)
    assert all(module.forward.__func__ is method for module, method in methods)
    snapshot = first.clone()
    torch.testing.assert_close(first, expected, atol=0.01, rtol=0.03)
    pixels.add_(0.1)
    second = runner(pixels)
    torch.testing.assert_close(second, vae.model.encode(pixels, vae.scale), atol=0.01, rtol=0.03)
    assert not torch.equal(first, second)
    vae.model.encoder.conv1.weight.add_(0.01)
    runner(pixels)
    assert runner.graph.captures == 2
    vae.scale[0].add_(0.2)
    final = runner(pixels)
    torch.testing.assert_close(final, runner._encode(pixels, vae.scale), atol=0, rtol=0)
    torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
    runner.close()
    assert all(module.forward.__func__ is method for module, method in methods)
    assert not any("_inference_" in name for name, _ in vae.named_buffers())
