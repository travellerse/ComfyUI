"""Routing tests for the low-VRAM fused-attention path (no GPU required).

Two separate claims, both CPU-testable:

1. The policy in `comfy/ldm/qwen_image21/fused_attention.py`: the score budget decides speed,
   and an unaffordable chunk decides feasibility. `prefer_fallback` must keep its measured
   behaviour when it is not told about memory, and must stop vetoing the kernel when the fallback
   cannot allocate the very chunk it is about to run.

2. The reachability fix in `attention_sub_quad`: the per-chunk `fused_attention` hook lives in
   `_get_attention_scores_no_kv_chunking`, which the greedy ladder stops using as soon as it can
   afford no chunk (`kv_chunk_size` stays None -> two-pass sqrt KV chunking). The whole call has to
   be offered to the backend before that path is chosen, or the kernel is unreachable exactly in the
   regime it was written for.
"""

import math

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.modules import attention as attn_mod  # noqa: E402
from comfy.ldm.qwen_image21 import fused_attention as fa  # noqa: E402

HEADS = 2
DIM = 4
BH = HEADS
Q_TOKENS = 2048
K_TOKENS = 512


def _qkv(dtype=torch.float16):
    q = torch.randn(BH, Q_TOKENS, DIM, dtype=dtype)
    k = torch.randn(BH, K_TOKENS, DIM, dtype=dtype)
    v = torch.randn(BH, K_TOKENS, DIM, dtype=dtype)
    return q, k.transpose(-1, -2), v


def _as_attention_args(dtype=torch.float16):
    """(b, heads, tokens, dim) as attention_sub_quad receives them with skip_reshape."""
    q = torch.randn(1, HEADS, Q_TOKENS, DIM, dtype=dtype)
    k = torch.randn(1, HEADS, K_TOKENS, DIM, dtype=dtype)
    v = torch.randn(1, HEADS, K_TOKENS, DIM, dtype=dtype)
    return q, k, v

# --- policy: pure functions -------------------------------------------------------------

def test_score_matrix_bytes_counts_the_chunk_not_the_whole_sequence():
    q, key_t, _ = _qkv()
    assert fa.score_matrix_bytes(q, key_t) == BH * Q_TOKENS * K_TOKENS * 2


def test_budget_rule_is_unchanged_when_memory_is_unknown():
    q, key_t, _ = _qkv()
    assert fa.prefer_fallback(q, key_t) is (fa.score_matrix_bytes(q, key_t) <= fa.SCORE_BUDGET_BYTES)
    assert fa.prefer_fallback(q, key_t, None) == fa.prefer_fallback(q, key_t)


def test_ample_memory_keeps_the_measured_fallback_preference():
    q, key_t, _ = _qkv()
    plenty = fa.score_matrix_bytes(q, key_t) * 100
    assert fa.prefer_fallback(q, key_t, plenty) is True


def test_unaffordable_chunk_serves_the_kernel_even_below_the_budget():
    q, key_t, _ = _qkv()
    scores = fa.score_matrix_bytes(q, key_t)
    assert scores <= fa.SCORE_BUDGET_BYTES  # the budget alone would prefer the fallback
    boundary = scores * fa.FALLBACK_POOL_FACTOR
    assert fa.prefer_fallback(q, key_t, boundary - 1) is False
    assert fa.prefer_fallback(q, key_t, boundary) is True


def test_the_affordability_boundary_is_the_measured_one_and_below_the_ladder():
    q, key_t, _ = _qkv()
    scores = fa.score_matrix_bytes(q, key_t)
    # Measured cliff (bench_starved_vram v2): the fallback still ran at 900 MiB priced free for a
    # 643 MiB chunk and died at 700 MiB, i.e. between 1.09x and 1.40x. 1.25x sits inside that band.
    assert 1.09 * scores < scores * fa.FALLBACK_POOL_FACTOR < 1.40 * scores
    # And it must stay well under the greedy ladder's 4x pricing, or the kernel would take over
    # geometry the fallback serves 1.4x faster.
    assert fa.prefer_fallback(q, key_t, scores * 3) is True
    assert fa.FALLBACK_POOL_FACTOR < 4.0


# --- policy: the hook the DiT actually passes -------------------------------------------

def test_guarded_hook_does_not_query_the_device_for_cpu_tensors(monkeypatch):
    """A serveable geometry off-device must still route, without paying a driver round trip."""
    from comfy import model_management

    def boom(*a, **kw):
        raise AssertionError("must not ask for device memory about a CPU tensor")

    monkeypatch.setattr(model_management, "get_free_memory", boom)
    q, key_t, v = _probe_shape()
    assert fa.needs_free_memory_probe(q, key_t, v, False, None) is True   # geometry wants the answer
    before = fa.stats["budget_fallbacks"]
    assert fa.fused_attention_guarded(q, key_t, v, 128 ** -0.5, False, None) is None
    assert fa.stats["budget_fallbacks"] == before + 1                      # decided by the budget alone


def test_guarded_hook_declines_masked_calls_without_touching_triton():
    q, key_t, v = _qkv()
    mask = torch.zeros(BH, Q_TOKENS, K_TOKENS)
    assert fa.fused_attention_guarded(q, key_t, v, DIM ** -0.5, False, mask) is None
    assert fa.stats["budget_fallbacks"] + fa.stats["fallback_calls"] > 0


# --- probe ordering: the driver query is the expensive part (65.7 us/call on gfx1030) --------

def _probe_shape(bh=2, q=256, k=512, dim=128):
    """fp16, head_dim 128: the geometry `_unsupported_reason` accepts. Score = bh*q*k*2 bytes."""
    query = torch.randn(bh, q, dim, dtype=torch.float16)
    key_t = torch.randn(bh, dim, k, dtype=torch.float16)
    value = torch.randn(bh, k, dim, dtype=torch.float16)
    return query, key_t, value


def test_probe_runs_only_for_served_geometry_below_the_budget():
    q, key_t, v = _probe_shape()
    with torch.no_grad():
        assert fa.needs_free_memory_probe(q, key_t, v, False, None) is True
        # above budget: the budget already says "serve", so the answer needs no driver read
        big_q, big_key, big_v = _probe_shape(bh=32, q=2048, k=10287)
        assert fa.score_matrix_bytes(big_q, big_key) > fa.SCORE_BUDGET_BYTES
        assert fa.needs_free_memory_probe(big_q, big_key, big_v, False, None) is False


def test_probe_is_skipped_for_geometry_the_kernel_could_not_serve_whatever_the_vram_is():
    q, key_t, v = _probe_shape()
    mask = torch.zeros(q.shape[0], q.shape[1], key_t.shape[2])
    head_dim_mismatch = _probe_shape(dim=64)
    with torch.no_grad():
        assert fa.needs_free_memory_probe(q, key_t, v, False, mask) is False
        assert fa.needs_free_memory_probe(q, key_t, v, True, None) is False
        assert fa.needs_free_memory_probe(*head_dim_mismatch, False, None) is False


def test_probe_gate_is_deliberately_blind_to_device_and_grad_state():
    """Routing bookkeeping must not change off-device, so those checks stay in the kernel."""
    q, key_t, v = _probe_shape()
    assert q.is_cuda is False                       # this suite runs with no visible device
    assert fa.needs_free_memory_probe(q, key_t, v, False, None) is True
    with torch.no_grad():
        assert fa.needs_free_memory_probe(q, key_t, v, False, None) is True   # same answer either way
    assert fa.fused_attention(q, key_t, v, 128 ** -0.5, False, None) is None  # the kernel still declines


def test_guarded_hook_ordering_does_not_regret_the_decision():
    """Reordering must not change *what* is returned, only who gets asked."""
    q, key_t, v = _probe_shape()
    with torch.no_grad():
        served = fa.fused_attention_guarded(q, key_t, v, 128 ** -0.5, False, None)
    assert served is None or served.shape == q.shape          # CPU: the kernel declines on device grounds
    masked = fa.fused_attention_guarded(q, key_t, v, 128 ** -0.5, False, torch.zeros(q.shape[0], q.shape[1], key_t.shape[2]))
    assert masked is None


# --- reachability: attention_sub_quad's ladder ------------------------------------------

class RecordingHook:
    """A backend that always answers, recording the query length of every call it gets."""

    def __init__(self, out_shape):
        self.calls = []
        self.out_shape = out_shape

    def __call__(self, query, key_t, value, scale, upcast_attention, mask):
        self.calls.append(query.shape[1])
        return torch.zeros(self.out_shape, dtype=query.dtype)

    @property
    def total_calls(self):
        return len(self.calls)


def _run_with_free_memory(monkeypatch, mem_free_bytes):
    from comfy import model_management

    monkeypatch.setattr(model_management, "get_free_memory", lambda dev=None, too=False: (mem_free_bytes, mem_free_bytes))
    q, k, v = _as_attention_args()
    hook = RecordingHook((BH, Q_TOKENS, DIM))
    out = attn_mod.attention_sub_quad(
        q, k, v, HEADS, mask=None,
        skip_reshape=True, skip_output_reshape=True, fused_attention=hook,
    )
    return hook, out


def test_starved_ladder_offers_the_whole_call_once(monkeypatch):
    # Cheaper than any chunk in the ladder for this k_tokens: forces kv_chunk_size to stay None.
    hook, out = _run_with_free_memory(monkeypatch, 1024)
    assert hook.calls == [Q_TOKENS], "the fallback must not split the query before asking the backend"
    assert out.shape == (1, HEADS, Q_TOKENS, DIM)


def test_affordable_ladder_chunks_the_query_and_keeps_the_old_behaviour(monkeypatch):
    # Enough for a 256-row chunk (256 * BH * bytes * 4 * k_tokens) but not for 512.
    need_256 = BH * 2 * 256 * 4.0 * K_TOKENS
    hook, out = _run_with_free_memory(monkeypatch, need_256 * 1.01)
    assert hook.total_calls > 1, "an affordable chunk ladder should still drive the per-chunk hook"
    assert max(hook.calls) <= 256
    assert hook.calls[0] * hook.total_calls >= Q_TOKENS
    assert math.isclose(sum(hook.calls), Q_TOKENS), "chunk lengths must tile the query"
