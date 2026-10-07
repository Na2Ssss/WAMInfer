"""BF16 masked, joint and request-local prefix attention kernels."""

import torch
import triton
import triton.language as tl


@torch.compiler.assume_constant_result
def _is_ada(device):
    return torch.cuda.get_device_capability(device) == (8, 9)


@torch.compiler.assume_constant_result
def _is_ampere80(device):
    return torch.cuda.get_device_capability(device) == (8, 0)


@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    MASK,
    OUT,
    QB: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KS: tl.constexpr,
    VB: tl.constexpr,
    VS: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TAIL: tl.constexpr,
    SKIP_MASKED: tl.constexpr,
):
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dim = tl.arange(0, D)
    query = tl.load(Q + batch * QB + rows[:, None] * QS + head * D + dim[None, :], rows[:, None] < NQ, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(NK // BN if TAIL else tl.cdiv(NK, BN)):
        keys = start * BN + cols
        active = True
        if SKIP_MASKED:
            # The key mask is shared across heads and query rows. Its current
            # value stays on the GPU, including during CUDA Graph replay.
            keep_keys = tl.load(MASK + batch * MB + keys * MK, keys < NK, 0)
            active = tl.sum(keep_keys.to(tl.int32), 0) > 0
        if active:
            key = tl.load(K + batch * KB + keys[None, :] * KS + head * D + dim[:, None], keys[None, :] < NK, 0)
            score = tl.dot(query, key) * (D**-0.5 * 1.4426950408889634)
            valid = keys[None, :] < NK
            if HAS_MASK:
                keep = tl.load(
                    MASK + batch * MB + head * MH + rows[:, None] * MQ + keys[None, :] * MK,
                    (rows[:, None] < NQ) & valid,
                    0,
                )
                valid = valid & keep
            score = tl.where(valid, score, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(score, 1))
            # An entirely masked tile must not poison later valid tiles (nor
            # all-masked query rows, for which SDPA returns zero).
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2(maximum - safe_max)
            probability = tl.exp2(score - safe_max[:, None])
            acc *= correction[:, None]
            value = tl.load(V + batch * VB + keys[:, None] * VS + head * D + dim[None, :], keys[:, None] < NK, 0)
            acc += tl.dot(probability.to(tl.bfloat16), value)
            normalizer = normalizer * correction + tl.sum(probability, 1)
            maximum = next_max
    if TAIL:
        # All 513 context positions are retained. Only the unused lanes of
        # the final hardware tile shrink from 128 to 16; no text is compacted.
        keys = NK - 1 + tl.arange(0, 16)
        key = tl.load(K + batch * KB + keys[None, :] * KS + head * D + dim[:, None], keys[None, :] < NK, 0)
        score = tl.dot(query, key) * (D**-0.5 * 1.4426950408889634)
        valid = keys[None, :] < NK
        if HAS_MASK:
            keep = tl.load(
                MASK + batch * MB + head * MH + rows[:, None] * MQ + keys[None, :] * MK, (rows[:, None] < NQ) & valid, 0
            )
            valid = valid & keep
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        probability = tl.exp2(score - safe_max[:, None])
        acc *= correction[:, None]
        value = tl.load(V + batch * VB + keys[:, None] * VS + head * D + dim[None, :], keys[:, None] < NK, 0)
        acc += tl.dot(probability.to(tl.bfloat16), value)
        normalizer = normalizer * correction + tl.sum(probability, 1)
    result = acc / tl.where(normalizer > 0.0, normalizer, 1.0)[:, None]
    tl.store(OUT + (batch * NQ + rows[:, None]) * HEADS * D + head * D + dim[None, :], result, rows[:, None] < NQ)


def _launch(q, k, v, mask, heads, bm, bn, warps=4, stages=2, tail=False, skip_masked=False):
    batch, nq, hidden = q.shape
    nk, d = k.shape[1], hidden // heads
    if mask is not None:
        mask = torch.broadcast_to(mask, (batch, heads, nq, nk))
        strides = mask.stride()
    else:
        strides = (0, 0, 0, 0)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    torch.library.wrap_triton(_attention_kernel)[(triton.cdiv(nq, bm), batch * heads)](
        q,
        k,
        v,
        mask,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        *strides,
        heads,
        d,
        nq,
        nk,
        mask is not None,
        bm,
        bn,
        tail and nk == 513 and bn == 128,
        skip_masked and mask is not None and strides[1] == 0 and strides[2] == 0,
        num_warps=warps,
        num_stages=stages,
    )
    return output


@triton.jit
def _prefix_context_attention(
    Q,
    K,
    V,
    MASK,
    OUT,
    QB: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KS: tl.constexpr,
    VB: tl.constexpr,
    VS: tl.constexpr,
    MB: tl.constexpr,
    MK: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    NQ: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    positions = tl.arange(0, 512)
    keep = tl.load(MASK + batch * MB + positions * MK)
    length = tl.sum(keep.to(tl.int32), 0)
    prefix = tl.sum((keep == (positions < length)).to(tl.int32), 0) == 512
    last = tl.load(MASK + batch * MB + 512 * MK)
    active = length + last.to(tl.int32)
    blocks = tl.cdiv(active, BN) if prefix else tl.cdiv(513, BN)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dim = tl.arange(0, D)
    query = tl.load(Q + batch * QB + rows[:, None] * QS + head * D + dim[None, :], rows[:, None] < NQ, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for index in range(blocks):
        virtual = index * BN + cols
        if prefix:
            keys = tl.where(virtual < length, virtual, 512)
            valid = virtual < active
        else:
            keys = virtual
            valid = keys < 513
            valid = valid & tl.load(MASK + batch * MB + keys * MK, valid, 0)
        key = tl.load(K + batch * KB + keys[None, :] * KS + head * D + dim[:, None], valid[None, :], 0)
        score = tl.dot(query, key) * (D**-0.5 * 1.4426950408889634)
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe)
        probability = tl.exp2(score - safe[:, None])
        acc *= correction[:, None]
        value = tl.load(V + batch * VB + keys[:, None] * VS + head * D + dim[None, :], valid[:, None], 0)
        acc += tl.dot(probability.to(tl.bfloat16), value)
        normalizer = normalizer * correction + tl.sum(probability, 1)
        maximum = next_max
    result = acc / tl.where(normalizer > 0.0, normalizer, 1.0)[:, None]
    tl.store(OUT + (batch * NQ + rows[:, None]) * HEADS * D + head * D + dim[None, :], result, rows[:, None] < NQ)


def _launch_prefix(q, k, v, mask, heads):
    batch, nq, hidden = q.shape
    mask = torch.broadcast_to(mask, (batch, heads, nq, 513))
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    torch.library.wrap_triton(_prefix_context_attention)[(triton.cdiv(nq, 32), batch * heads)](
        q,
        k,
        v,
        mask,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        mask.stride(0),
        mask.stride(3),
        heads,
        hidden // heads,
        nq,
        32,
        64,
        num_warps=4,
        num_stages=2,
    )
    return output


@torch.library.triton_op("wam_infer::masked_attention_mask_tiles_v4", mutates_args={})
def masked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, heads: int, tail: bool = False
) -> torch.Tensor:
    if not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Triton attention requires CUDA BF16 inputs.")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("Attention expects matching BSH inputs.")
    if heads < 1 or q.shape[-1] != k.shape[-1] or q.shape[-1] // heads != 128:
        raise ValueError("Triton attention requires head dimension 128.")
    if q.shape[-1] % heads or min(q.shape[:2]) < 1 or k.shape[1] < 1:
        raise ValueError("Attention requires nonempty input and complete heads.")
    if any(x.device != q.device or x.stride(-1) != 1 for x in (k, v, q)):
        raise ValueError("Attention requires matching devices and contiguous channels.")
    if mask is not None and (mask.dtype != torch.bool or mask.device != q.device):
        raise ValueError("Attention mask must be boolean and on the input device.")
    if k.shape[1] == 513 and heads == 24 and mask is not None and _is_ampere80(q.device):
        broadcast = torch.broadcast_to(mask, (q.shape[0], heads, q.shape[1], 513))
        if broadcast.stride(1) == 0 and broadcast.stride(2) == 0:
            return _launch_prefix(q, k, v, mask, heads)
    bm = 16 if q.shape[1] <= 32 else 32
    bn = 128 if k.shape[1] >= 512 else 64
    stages = 3 if q.shape[1] >= 360 and k.shape[1] < 512 else 2
    # Ada's smaller shared-memory budget favors one pipeline stage for this
    # dense Video/context shape. Tiles, masks and softmax arithmetic stay fixed.
    if q.shape[1] == 240 and k.shape[1] == 513 and heads == 24 and _is_ada(q.device):
        stages = 1
    if heads == 24 and _is_ampere80(q.device):
        if q.shape[1] == 360 and k.shape[1] == 513:
            bm, bn, stages = 16, 128, 2
        elif q.shape[1] == 392 and k.shape[1] == 392:
            bm, bn, stages = 64, 64, 2
        elif q.shape[1] == 32 and k.shape[1] == 513:
            stages = 1
    # Keep the measured optimization local to OpenWAM's Ada context shape.
    # Other masks/hardware use the original dense loop; no CPU mask inspection.
    skip_masked = k.shape[1] == 513 and heads == 24 and (_is_ada(q.device) or _is_ampere80(q.device))
    return _launch(q, k, v, mask, heads, bm, bn, stages=stages, tail=tail, skip_masked=skip_masked)


@triton.jit
def _joint_attention(
    Q,
    K,
    V,
    OUT,
    NQ: tl.constexpr,
    VIDEO: tl.constexpr,
    FIRST: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    group = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    clean_groups = tl.cdiv(FIRST, BM)
    clean = group < clean_groups
    row_start = tl.where(clean, group * BM, FIRST + (group - clean_groups) * BM)
    rows = row_start + tl.arange(0, BM)
    row_valid = rows < tl.where(clean, FIRST, NQ)
    columns = tl.arange(0, BN)
    dim = tl.arange(0, D)
    query = tl.load(Q + (batch * NQ + rows[:, None]) * HEADS * D + head * D + dim[None, :], row_valid[:, None], 0)
    # Native mutual masking isolates clean-frame queries from BOTH future
    # Video and Action keys. All clean-frame queries are still recomputed.
    key_count = tl.where(clean, FIRST, NQ)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    for index in range(tl.cdiv(key_count, BN)):
        virtual = index * BN + columns
        keys = virtual
        valid = virtual < key_count
        key = tl.load(K + (batch * NQ + keys[None, :]) * HEADS * D + head * D + dim[:, None], valid[None, :], 0)
        scores = tl.dot(query, key) * (D**-0.5 * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp2(maximum - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        accumulator *= correction[:, None]
        value = tl.load(V + (batch * NQ + keys[:, None]) * HEADS * D + head * D + dim[None, :], valid[:, None], 0)
        accumulator += tl.dot(probabilities.to(tl.bfloat16), value)
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        maximum = new_max
    result = accumulator / normalizer[:, None]
    tl.store(OUT + (batch * NQ + rows[:, None]) * HEADS * D + head * D + dim[None, :], result, row_valid[:, None])


def _launch_joint(q, k, v, video, first, heads, bm=32, bn=64, warps=4, stages=2):
    batch, count, width = q.shape
    output = torch.empty_like(q)
    grid = (triton.cdiv(first, bm) + triton.cdiv(count - first, bm), batch * heads)
    torch.library.wrap_triton(_joint_attention)[grid](
        q, k, v, output, count, video, first, heads, width // heads, bm, bn, num_warps=warps, num_stages=stages
    )
    return output


@torch.library.triton_op("wam_infer::joint_first_frame_attention_v2", mutates_args={})
def joint_first_frame_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, video: int, first: int, heads: int
) -> torch.Tensor:
    if torch.is_grad_enabled() or not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("Joint attention requires CUDA BF16 inference.")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3 or q.shape[-1] != heads * 128:
        raise ValueError("Joint attention requires matching BSH tensors with head_dim=128.")
    if not 0 < first <= video < q.shape[1] or not all(t.is_contiguous() and t.device == q.device for t in (q, k, v)):
        raise ValueError("Joint attention requires valid Video boundaries and contiguous inputs.")
    ada = _is_ada(q.device)
    return _launch_joint(
        q, k, v, video, first, heads, bm=32 if ada else 128, bn=64 if ada else 128, warps=4 if ada else 8, stages=1
    )


@triton.jit
def _split_attention_kernel(
    QV,
    QA,
    KC,
    KV,
    KA,
    VC,
    VV,
    VA,
    OV,
    OA,
    QVB: tl.constexpr,
    QVS: tl.constexpr,
    QAB: tl.constexpr,
    QAS: tl.constexpr,
    KCB: tl.constexpr,
    KCS: tl.constexpr,
    KVB: tl.constexpr,
    KVS: tl.constexpr,
    KAB: tl.constexpr,
    KAS: tl.constexpr,
    VCB: tl.constexpr,
    VCS: tl.constexpr,
    VVB: tl.constexpr,
    VVS: tl.constexpr,
    VAB: tl.constexpr,
    VAS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    NA: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dim = tl.arange(0, D)
    nq: tl.constexpr = NV
    qp = tl.where(
        rows[:, None] < nq,
        QV + batch * QVB + rows[:, None] * QVS + head * D + dim[None, :],
        QA + batch * QAB + (rows[:, None] - nq) * QAS + head * D + dim[None, :],
    )
    query = tl.load(qp, rows[:, None] < nq + NA, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(NC + NV + NA, BN)):
        keys = start * BN + cols
        kp = tl.where(
            keys[None, :] < NC,
            KC + batch * KCB + keys[None, :] * KCS + head * D + dim[:, None],
            tl.where(
                keys[None, :] < NC + NV,
                KV + batch * KVB + (keys[None, :] - NC) * KVS + head * D + dim[:, None],
                KA + batch * KAB + (keys[None, :] - NC - NV) * KAS + head * D + dim[:, None],
            ),
        )
        key = tl.load(kp, keys[None, :] < NC + NV + NA, 0)
        score = tl.dot(query, key) * (D**-0.5 * 1.4426950408889634)
        score = tl.where(keys[None, :] < NC + NV + NA, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        probability = tl.exp2(score - safe_max[:, None])
        acc *= correction[:, None]
        vp = tl.where(
            keys[:, None] < NC,
            VC + batch * VCB + keys[:, None] * VCS + head * D + dim[None, :],
            tl.where(
                keys[:, None] < NC + NV,
                VV + batch * VVB + (keys[:, None] - NC) * VVS + head * D + dim[None, :],
                VA + batch * VAB + (keys[:, None] - NC - NV) * VAS + head * D + dim[None, :],
            ),
        )
        value = tl.load(vp, keys[:, None] < NC + NV + NA, 0)
        acc += tl.dot(probability.to(tl.bfloat16), value)
        normalizer = normalizer * correction + tl.sum(probability, 1)
        maximum = next_max
    result = acc / tl.where(normalizer > 0.0, normalizer, 1.0)[:, None]
    op = tl.where(
        rows[:, None] < nq,
        OV + (batch * nq + rows[:, None]) * H * D + head * D + dim[None, :],
        OA + (batch * NA + rows[:, None] - nq) * H * D + head * D + dim[None, :],
    )
    tl.store(op, result, rows[:, None] < nq + NA)


@torch.library.triton_op("wam_infer::current_frame_attention_v1", mutates_args={})
def current_frame_attention(
    qv: torch.Tensor,
    qa: torch.Tensor,
    ck: torch.Tensor,
    kv: torch.Tensor,
    ka: torch.Tensor,
    cv: torch.Tensor,
    vv: torch.Tensor,
    va: torch.Tensor,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = (qv, qa, ck, kv, ka, cv, vv, va)
    if any(t.ndim != 3 or t.device != qv.device or t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in tensors):
        raise ValueError("Split attention requires CUDA BF16 BSH inputs with contiguous channels.")
    if not qv.is_cuda or heads < 1 or qv.shape[-1] != heads * 128:
        raise ValueError("Split attention requires CUDA and head dimension 128.")
    if any(t.shape[0] != qv.shape[0] or t.shape[-1] != qv.shape[-1] or min(t.shape[:2]) < 1 for t in tensors):
        raise ValueError("Split attention requires matching batch/channels and nonempty banks.")
    if (
        qv.shape[1] != kv.shape[1]
        or kv.shape != vv.shape
        or qa.shape != ka.shape
        or qa.shape != va.shape
        or ck.shape != cv.shape
    ):
        raise ValueError("Split attention requires matching query/key/value banks per modality.")
    ov = torch.empty(qv.shape, device=qv.device, dtype=qv.dtype)
    oa = torch.empty(qa.shape, device=qa.device, dtype=qa.dtype)
    nc, nv, na = ck.shape[1], kv.shape[1], qa.shape[1]
    nq = qv.shape[1]
    bm = 16 if nq + na <= 32 else 32
    bn = 128 if nc + nv + na >= 512 else 64
    stages = 2
    torch.library.wrap_triton(_split_attention_kernel)[(triton.cdiv(nq + na, bm), qv.shape[0] * heads)](
        qv,
        qa,
        ck,
        kv,
        ka,
        cv,
        vv,
        va,
        ov,
        oa,
        qv.stride(0),
        qv.stride(1),
        qa.stride(0),
        qa.stride(1),
        ck.stride(0),
        ck.stride(1),
        kv.stride(0),
        kv.stride(1),
        ka.stride(0),
        ka.stride(1),
        cv.stride(0),
        cv.stride(1),
        vv.stride(0),
        vv.stride(1),
        va.stride(0),
        va.stride(1),
        heads,
        128,
        nc,
        nv,
        na,
        bm,
        bn,
        num_warps=4,
        num_stages=stages,
    )
    return ov, oa
