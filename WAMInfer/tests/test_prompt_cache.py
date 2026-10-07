import pytest
import torch
from WAMInfer.graphs import TextEncodeGraph
from openwam.model.video_backbone.wan.models.text_encoder import WanTextEncoder


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_prompt_cache_refreshes_for_prompt_weights_and_eval_mode():
    torch.manual_seed(42)
    encoder = (
        WanTextEncoder(
            vocab=32, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2, num_buckets=16, shared_pos=False
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    graph = TextEncodeGraph(encoder)
    calls = []

    def tokenizer(prompts, **kw):
        calls.append(prompts[0])
        ids = torch.zeros(1, 512, dtype=torch.long)
        ids[:, :18] = 1 if prompts[0] == "first" else 2
        mask = torch.zeros_like(ids)
        mask[:, :18] = 1
        return (ids, mask)

    def encode(prompt):
        return graph.encode_prompt(prompt, tokenizer, "cuda")[0]

    first = encode("first").clone()
    torch.testing.assert_close(encode("first"), first, atol=0, rtol=0)
    assert calls == ["first"]
    other = encode("second").clone()
    assert not torch.equal(other, first)
    torch.testing.assert_close(encode("first"), first, atol=0, rtol=0)
    encoder.token_embedding.weight.add_(0.2)
    changed = encode("first")
    assert not torch.equal(changed, first)
    assert calls == ["first", "second", "first", "first"]
    encoder.blocks[0].train()
    with pytest.raises(ValueError, match="eval"):
        encode("first")
    graph.close()
