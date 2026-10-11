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

# The kernel is forward only and carries no autograd, so it is only correct under
# inference. Exposed so tests and the benchmark harness can assert what actually ran.
stats = {"fused_calls": 0, "fallback_calls": 0, "compile_failures": 0}
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
