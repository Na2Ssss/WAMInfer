"""Fused Q/K RMSNorm, RoPE, video modulation and residual normalization."""

import torch
import triton
import triton.language as tl


@torch.compiler.assume_constant_result
def _is_ada(device):
    return torch.cuda.get_device_capability(device) == (8, 9)


@triton.jit
def _qk_rms_rope_kernel(
    Q,
    K,
    WQ,
    WK,
    FREQ,
    OQ,
    OK,
    S: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    QB: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KS: tl.constexpr,
    FS: tl.constexpr,
    FP: tl.constexpr,
    FC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0) // tl.cdiv(H, CHUNK)
    chunk = tl.program_id(0) % tl.cdiv(H, CHUNK)
    is_key = tl.program_id(1) == 1
    batch, token = row // S, row % S
    source = tl.where(is_key, K, Q)
    weight = tl.where(is_key, WK, WQ)
    output = tl.where(is_key, OK, OQ)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < H
    base = batch * tl.where(is_key, KB, QB) + token * tl.where(is_key, KS, QS)
    values = tl.load(source + base + offsets, valid, 0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, 0) / H + EPS)
    if CHUNK != BLOCK:
        # Repeat the unchanged row reduction, then rotate only this channel
        # chunk. Ada avoids FP64 work on padded channels and uses smaller
        # live tensors without changing the reduction or rounding order.
        offsets = chunk * CHUNK + tl.arange(0, CHUNK)
        valid = offsets < H
        values = tl.load(source + base + offsets, valid, 0).to(tl.float32)
    scale = tl.load(weight + offsets, valid, 0).to(tl.float32)
    # These are the native RMSNorm's two BF16 rounding boundaries. Explicit
    # float32 casts prevent the compiler from discarding either boundary.
    normalized = (values * inverse_rms).to(tl.bfloat16).to(tl.float32)
    normalized = (normalized * scale).to(tl.bfloat16)
    even, odd = tl.split(normalized.reshape(CHUNK // 2, 2))
    paired = tl.join(odd, even).reshape(CHUNK)
    position = token * FS + ((offsets % HD) // 2) * FP
    cosine = tl.load(FREQ + position, valid, 1).to(tl.float64)
    sine = tl.load(FREQ + position + FC, valid, 0).to(tl.float64)
    a, b = normalized.to(tl.float64), paired.to(tl.float64)
    result = tl.where(offsets % 2 == 0, a * cosine - b * sine, a * cosine + b * sine)
    tl.store(output + row * H + offsets, result, valid)


# Version the decomposed op when its implementation changes. Inductor's
# persistent FX cache can otherwise retain the old generated Triton schedule.
@torch.library.triton_op("wam_infer::qk_rms_rope_v2", mutates_args={})
def rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    frequency: torch.Tensor,
    num_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, hidden = q.shape
    frequency = frequency.reshape(seq, hidden // num_heads // 2, 2)
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty_like(oq)
    block = triton.next_power_of_2(hidden)
    chunk = 1024 if hidden == 3072 and _is_ada(q.device) else block
    torch.library.wrap_triton(_qk_rms_rope_kernel)[(batch * seq * triton.cdiv(hidden, chunk), 2)](
        q,
        k,
        wq,
        wk,
        frequency,
        oq,
        ok,
        S=seq,
        H=hidden,
        HD=hidden // num_heads,
        QB=q.stride(0),
        QS=q.stride(1),
        KB=k.stride(0),
        KS=k.stride(1),
        FS=frequency.stride(0),
        FP=frequency.stride(1),
        FC=frequency.stride(2),
        EPS=eps,
        BLOCK=block,
        CHUNK=chunk,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return oq, ok


# Stable values for the existing Triton MODE specializations. The names let
# callers show which residual boundary is being fused without changing math.
SELF_INPUT = 0
SELF_RESIDUAL = 1
CROSS_RESIDUAL = 2
FFN_RESIDUAL = 3


@triton.jit
def _modulation(MOD, TIME, row, d, slot: tl.constexpr, H: tl.constexpr, TS: tl.constexpr):
    a = tl.load(MOD + slot * H + d, d < H, 0).to(tl.float32)
    b = tl.load(TIME + row * TS + slot * H + d, d < H, 0).to(tl.float32)
    return (a + b).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _modulated_norm(
    X,
    UPDATE,
    MOD,
    PREVIOUS,
    TIME,
    W,
    BIAS,
    Y,
    NORM,
    H: tl.constexpr,
    XS: tl.constexpr,
    US: tl.constexpr,
    TS: tl.constexpr,
    MODE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    valid = d < H
    x = tl.load(X + row * XS + d, valid, 0).to(tl.float32)
    if MODE > 0:
        update = tl.load(UPDATE + row * US + d, valid, 0).to(tl.float32)
        if MODE == 1:
            gate = _modulation(MOD, TIME, row, d, 2, H, TS)
            update = (gate * update).to(tl.bfloat16).to(tl.float32)
        elif MODE == 3:
            gate = _modulation(PREVIOUS, TIME, row, d, 5, H, TS)
            update = (gate * update).to(tl.bfloat16).to(tl.float32)
        x = (x + update).to(tl.bfloat16).to(tl.float32)
    centered = x - tl.sum(x, 0) / H
    variance = tl.sum(tl.where(valid, centered * centered, 0.0), 0) / H
    normalized = centered * tl.rsqrt(variance + EPS)
    if MODE == 1:
        weight = tl.load(W + d, valid, 0).to(tl.float32)
        bias = tl.load(BIAS + d, valid, 0).to(tl.float32)
        normalized = normalized * weight + bias
    else:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
        shift = _modulation(MOD, TIME, row, d, 3 if MODE == 2 else 0, H, TS)
        scale = _modulation(MOD, TIME, row, d, 4 if MODE == 2 else 1, H, TS)
        scale = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
        normalized = (normalized * scale).to(tl.bfloat16).to(tl.float32) + shift
    tl.store(Y + row * H + d, x, valid)
    tl.store(NORM + row * H + d, normalized, valid)


@torch.library.triton_op("wam_infer::modulated_norm", mutates_args={})
def modulated_norm(
    x: torch.Tensor,
    update: torch.Tensor | None,
    mod: torch.Tensor,
    previous: torch.Tensor | None,
    time: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mode: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the updated residual and the normalized input to the next GEMM.

    SELF_INPUT normalizes the initial state; SELF_RESIDUAL includes gated
    self-attention output; CROSS_RESIDUAL adds cross-attention output;
    FFN_RESIDUAL folds the previous block's gated FFN into the next block.
    """
    y, norm = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device),
        torch.empty(x.shape, dtype=x.dtype, device=x.device),
    )
    torch.library.wrap_triton(_modulated_norm)[(x.shape[1],)](
        x,
        update,
        mod,
        previous,
        time,
        weight,
        bias,
        y,
        norm,
        x.shape[-1],
        x.stride(1),
        0 if update is None else update.stride(1),
        time.stride(1) if time.ndim == 4 and time.shape[1] != 1 else 0,
        mode,
        eps,
        triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y, norm
