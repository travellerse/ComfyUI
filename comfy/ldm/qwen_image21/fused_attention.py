"""Triton fused online-softmax attention for the Qwen-Image-2.1 DiT.

The `attention_sub_quad` path materialises two `(batch_x_heads, q_chunk, k_tokens)`
tensors per query chunk: the raw scores from `baddbmm` and a separate probability
tensor from the out-of-place softmax. At the pinned `query_chunk_size=1024` that is
768 MiB of transient VRAM at 1024x1536, and it is the peak of the whole attention
block. Neither tensor is needed: the block update below keeps the running max, the
running denominator and the output accumulator, and never writes a score matrix.

This is deliberately narrow. It handles the mask-free fp16 image attention of this
one model on RDNA2, and returns `None` for everything else so the caller falls
back to the existing implementation. Support is decided by inspecting the tensors,
not by catching exceptions, so a bug in this file surfaces as a crash rather than
as a silent downgrade.

Correction after measurement (evidence/ reports 42 and 43): the transient the docstring above
calls "the peak of the whole attention block" is not the peak of the *run*. The sub-quadratic path
allocates and frees it inside the attention call, and the block's peak is set by the MLP's
`gate_up` output, so both paths peak within ~2 MiB of each other at every geometry tested here --
while the sub-quadratic path is 1.46x faster. Hence `SCORE_BUDGET_BYTES` and
`fused_attention_guarded` below: the model passes the guarded entry point, so this kernel is the
large-geometry branch rather than the default. `fused_attention` itself keeps its original
"can this be served" contract, which the device tests depend on.
"""

import logging

import torch

from comfy import model_management

# Validated on gfx1031 (reported gfx1030 under HSA_OVERRIDE_GFX_VERSION) with ROCm
# 7.2.53211 and Triton 3.7.1. BLOCK_N=128 is a hard limit on this family, not a
# tuning preference: it needs 81920 bytes of shared memory and the card has less.
# These stay module constants rather than parameters so no caller can trigger a
# combinatorial compile sweep.
BLOCK_M = 128
BLOCK_N = 64
NUM_WARPS = 8
NUM_STAGES = 2

# HEAD_DIM is a constexpr, so every distinct value is a separate compilation.
# Qwen-Image-2.1 fixes attention_head_dim=128, and restricting to it keeps the
# steady state at one specialization.
SUPPORTED_HEAD_DIM = 128

# gfx1031 is the only part measured. The rest of the RDNA2 family shares its shared
# memory ceiling, but that is an assumption, so they are listed and not claimed.
SUPPORTED_ARCHS = ("gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036")

# When this kernel should be used at all, as opposed to when it merely *can* be.
#
# Measured on gfx1030 through the same entry point the DiT uses, at the production geometry
# (batch_x_heads=32, query_chunk_size=1024, k_tokens=10287 -> 642 MiB of scores): the sub-quadratic
# path is 1.46x faster (115.75 vs 168.69 ms per block call), and its score tensors are served from
# the pool the allocator had already reserved, so the run's peak physical VRAM does not move
# (10922-11336 MiB ON vs 10970-11273 MiB OFF, report 43 §4.1; per-block peak 752 vs 720 MiB at
# 5120 tokens and 2116 vs 2114 MiB at 15000 tokens, report 43 §2.3). The saving this kernel was
# written for is therefore real at the allocation level and invisible at the peak.
#
# Above this boundary nothing has been measurable on this card: the 2400x1600 branch reaches
# 12120 MiB of 12272 with the fused path and was interrupted by the host-RAM guard before it could
# finish in either mode (report 43 §3). So keeping the kernel there is a margin, not a demonstrated
# benefit -- and it must not be described as a proven OOM guard.
#
# The budget is not the whole rule. When the fallback cannot afford even its smallest query chunk it
# moves to two-pass sqrt(k_tokens) KV chunking, and `fused_attention` is not on that path at all, so
# there the kernel is offered the call first regardless of score size. That regime is the one
# bench/bench_starved_vram.py measures; until it reports, the starvation branch is a hypothesis with
# a mechanism, not a measured win.
SCORE_BUDGET_BYTES = 1024 ** 3


def score_matrix_bytes(query, key_t):
    """What the fallback would materialise for this call: one (BH, q_tokens, k_tokens) tensor."""
    return query.shape[0] * query.shape[1] * key_t.shape[2] * query.element_size()


# What the fallback really needs on top of one score tensor before it can run the chunk: measured
# on gfx1030 (bench/bench_starved_vram.py v2, BH=32 q=1024 k=10287, 643 MiB of scores) as a cliff
# between 900 MiB priced-free, where the sub-quadratic chunk still ran at 22.1 ms, and 700 MiB,
# where the same call raised OutOfMemoryError while this kernel ran the identical geometry in
# 31.8 ms. Anything in that band is guesswork, so the boundary sits inside it rather than on the
# optimistic end: 1.25x was chosen to make the surviving call, not to win the speed race.
FALLBACK_POOL_FACTOR = 1.25


def sub_quad_cannot_afford_chunk(query, key_t, mem_free_bytes):
    """Whether the fallback could not allocate the score matrix this chunk is about to build.

    `attention_sub_quad` gets `mem_free_bytes` from `model_management.get_free_memory()`, which
    counts driver-free *plus* torch's reserved-but-unallocated pool -- so this is false whenever the
    pool can serve the chunk, which is exactly the warm-pool case that measured the fallback as the
    faster one. The factor is the measured one above, not the greedy ladder's 4x (ladder:
    comfy/ldm/modules/attention.py, `batch_x_heads * bytes * chunk * 4.0 * k_tokens`): at 4x the
    kernel would take over geometry the fallback serves faster.
    """
    return mem_free_bytes < score_matrix_bytes(query, key_t) * FALLBACK_POOL_FACTOR


def prefer_fallback(query, key_t, mem_free_bytes=None):
    """Whether the sub-quadratic path should handle this call.

    Shape-based unless the caller also says how much VRAM the fallback has: a fallback that cannot
    allocate the chunk it is about to run is not the cheaper one, whatever the score budget says.
    """
    if mem_free_bytes is not None and sub_quad_cannot_afford_chunk(query, key_t, mem_free_bytes):
        return False
    return score_matrix_bytes(query, key_t) <= SCORE_BUDGET_BYTES


# The kernel is forward only and carries no autograd, so it is only correct under
# inference. Exposed so tests and the benchmark harness can assert what actually ran.
# `budget_fallbacks` counts calls this file could serve but hands back to the fallback because
# that is faster there -- see SCORE_BUDGET_BYTES.
stats = {"fused_calls": 0, "fallback_calls": 0, "compile_failures": 0, "budget_fallbacks": 0}
_logged_unsupported = set()

_triton_state = None


def _load_triton():
    """Import triton and build the kernel, or return `(None, reason)` if impossible.

    Deferred to first use: importing triton costs real time at startup and the
    default path must not pay for a backend it may never use.
    """
    global _triton_state
    if _triton_state is not None:
        return _triton_state

    if not model_management.is_amd() or torch.version.hip is None:
        _triton_state = (None, "not an AMD ROCm device")
        return _triton_state
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        _triton_state = (None, "could not read the device architecture")
        return _triton_state
    if arch not in SUPPORTED_ARCHS:
        _triton_state = (None, f"architecture {arch} is outside the supported set")
        return _triton_state

    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        _triton_state = (None, f"triton is unavailable: {e}")
        return _triton_state

    @triton.jit
    def _fwd(Q, K, V, Out, sm_scale, sqb, sqm, sqd, skb, skn, skd, svb, svn, svd, sob, som, sod,
             M, N, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # Online softmax, written out from the block update rather than ported from a
        # CUDA flash-attention source. exp2 with log2(e) folded into the scale keeps
        # the inner loop a single tl.dot chain:
        #   m_new = max(m, block_max)
        #   l_new = l * exp2(m - m_new) + block_sum * exp2(block_max - m_new)
        #   acc   = acc * exp2(m - m_new) + P @ V
        #   out   = acc / l
        start_m = tl.program_id(0)
        bh = tl.program_id(1)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + bh * sqb + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.4426950408889634

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            k_ptrs = K + bh * skb + offs_n[None, :] * skn + offs_d[:, None] * skd
            k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * qk_scale
            # A zero-filled padding load produces a zero score, which would survive
            # the running max and enter the denominator as real probability mass.
            # Force the tail of the last tile to -inf so it contributes nothing. The
            # loop bound guarantees every tile has a live column, so the max is never
            # -inf and exp2 stays finite.
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            v_ptrs = V + bh * svb + offs_n[:, None] * svn + offs_d[None, :] * svd
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc = acc * alpha[:, None]
            # fp16 operands with an fp32 accumulator. An fp32 PV dot doubles the
            # register pressure of the second dot and measured 4.5x slower.
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_ij

        acc = acc / l_i[:, None]
        o_ptrs = Out + bh * sob + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)

    _triton_state = ((triton, _fwd), None)
    return _triton_state


def _unsupported_reason(query, key_t, value, upcast_attention, mask):
    """Why this call cannot use the fused kernel, or `None` when it can."""
    if mask is not None:
        return "the call carries a mask"
    if upcast_attention:
        return "attention is upcast to fp32"
    if not query.is_cuda:
        return "tensors are not on a GPU"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if query.dtype != torch.float16 or key_t.dtype != torch.float16 or value.dtype != torch.float16:
        return f"dtype is not uniformly fp16 ({query.dtype}, {key_t.dtype}, {value.dtype})"
    if query.shape[0] != key_t.shape[0] or query.shape[0] != value.shape[0]:
        return "batch_x_heads disagree between query, key and value"
    if query.shape[2] != key_t.shape[1] or query.shape[2] != value.shape[2]:
        return "head_dim disagrees between query, key and value"
    if query.shape[1] == 0 or key_t.shape[2] == 0:
        return "empty query or key"
    if query.shape[2] != SUPPORTED_HEAD_DIM:
        return f"head_dim {query.shape[2]} is not the supported {SUPPORTED_HEAD_DIM}"
    return None


def _log_unsupported(reason):
    # Once per distinct reason. The masked text segment falls back on every single
    # run, so this must not repeat per call.
    if reason not in _logged_unsupported:
        _logged_unsupported.add(reason)
        logging.debug("Qwen-Image-2.1 fused attention not used: %s", reason)


def fused_attention(query, key_t, value, scale, upcast_attention, mask):
    """Fused attention for one query chunk, or `None` when it is not supported.

    `query` is `(batch_x_heads, q_tokens, head_dim)`, `key_t` is
    `(batch_x_heads, head_dim, k_tokens)` and `value` is
    `(batch_x_heads, k_tokens, head_dim)`, matching what
    `_get_attention_scores_no_kv_chunking` receives. The result has the shape and
    dtype of `query`.
    """
    reason = _unsupported_reason(query, key_t, value, upcast_attention, mask)
    if reason is not None:
        stats["fallback_calls"] += 1
        _log_unsupported(reason)
        return None

    loaded, load_reason = _load_triton()
    if loaded is None:
        stats["fallback_calls"] += 1
        _log_unsupported(load_reason)
        return None
    triton, kernel = loaded

    batch_x_heads, q_tokens, head_dim = query.shape
    k_tokens = key_t.shape[2]
    # key_t is (BH, D, N) and the kernel indexes K as (BH, N, D). Transposing is a
    # view, not a copy, and the kernel takes explicit strides, so the permuted key
    # layout never has to be materialised.
    k = key_t.transpose(-1, -2)
    out = torch.empty((batch_x_heads, q_tokens, head_dim), dtype=query.dtype, device=query.device)

    try:
        grid = (triton.cdiv(q_tokens, BLOCK_M), batch_x_heads)
        kernel[grid](
            query, k, value, out, scale,
            query.stride(0), query.stride(1), query.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            q_tokens, k_tokens,
            HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
    except Exception as e:
        # A kernel that does not fit this GPU's registers or shared memory is a
        # hardware limit, not a defect, and the caller has a correct fallback. Every
        # other failure (a shape bug, a typo) is a defect and must not be hidden.
        # Walked defensively: this lookup runs on the failure path, and an AttributeError
        # from it would replace the real exception with a confusing one.
        resource_error = getattr(getattr(getattr(triton, "runtime", None), "errors", None), "OutOfResources", None)
        if resource_error is not None and isinstance(e, resource_error):
            stats["compile_failures"] += 1
            _log_unsupported(f"the kernel does not fit this device: {e}")
            return None
        raise

    stats["fused_calls"] += 1
    if stats["fused_calls"] == 1:
        logging.debug("Qwen-Image-2.1 attention backend: triton_fused")
    return out


def is_supported(query, key_t, value, upcast_attention=False, mask=None):
    """Whether `fused_attention` would handle this call. For tests and diagnostics."""
    if _unsupported_reason(query, key_t, value, upcast_attention, mask) is not None:
        return False
    return _load_triton()[0] is not None


def fused_attention_guarded(query, key_t, value, scale, upcast_attention, mask):
    """The hook the DiT passes: `fused_attention`, except where the fallback is the better call.

    Deliberately a wrapper rather than a rule inside `fused_attention`: that function's contract is
    "can this shape be served by this kernel", which its device tests and the benchmark harness
    depend on. Whether it *should* be served is routing policy, and it belongs at the call site.
    """
    mem_free = model_management.get_free_memory(query.device, True)[0] if query.is_cuda else None
    if prefer_fallback(query, key_t, mem_free):
        stats["budget_fallbacks"] += 1
        return None
    return fused_attention(query, key_t, value, scale, upcast_attention, mask)
