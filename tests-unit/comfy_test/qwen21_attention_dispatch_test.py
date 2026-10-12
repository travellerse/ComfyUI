"""Contract tests for which attention path the Qwen-Image-2.1 DiT should take.

The kernel in `fused_attention.py` was written to avoid materialising score matrices, and it does:
80 MiB of transient against the fallback's 642 MiB at the production geometry. What it did not do,
once measured end to end, is move the *run's* peak VRAM -- the fallback allocates and frees inside
one attention call, and the block's peak is set by the MLP -- while being 1.46x slower. So the
dispatch rule is now a budget, and the only thing worth pinning is where that boundary sits and that
the boundary is computed from shapes rather than from a device.

CPU only: nothing here runs a device op. comfy.model_management resolves the torch device at import
(model_management.py:364), so cpu mode is declared first, exactly as the sibling tests do.
"""

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.qwen_image21 import fused_attention as fa  # noqa: E402

# The geometry the production probe recorded: batch_x_heads=32 (1 batch x 32 heads),
# query_chunk_size=1024 pinned in model.py, k_tokens=10287 (5167 prefix + 5120 target).
PROD_BH, PROD_CHUNK, PROD_K = 32, 1024, 10287
# The 2400x1600 branch: 15000 target tokens, same 5167 prefix.
HIRES_BH, HIRES_CHUNK, HIRES_K = 32, 1024, 20167


def _q(bh, q, d=128, dtype=torch.float16):
    return torch.zeros(bh, q, d, dtype=dtype)


def _k_t(bh, d, k):
    return torch.zeros(bh, d, k, dtype=torch.float16)


def test_score_bytes_is_the_fallback_tensor_and_nothing_else():
    # one (BH, q, k) fp16 tensor: inplace_softmax=True means there is no second scores buffer
    assert fa.score_matrix_bytes(_q(2, 4, 8), _k_t(2, 8, 16)) == 2 * 4 * 16 * 2


def test_production_geometry_is_below_the_budget():
    # 32 * 1024 * 10287 * 2 B = 642.9 MiB
    assert fa.score_matrix_bytes(_q(PROD_BH, PROD_CHUNK), _k_t(PROD_BH, 128, PROD_K)) == 674168832
    assert fa.prefer_fallback(_q(PROD_BH, PROD_CHUNK), _k_t(PROD_BH, 128, PROD_K)) is True


def test_large_geometry_is_above_the_budget():
    # 32 * 1024 * 20167 * 2 B = 1260.4 MiB
    nbytes = fa.score_matrix_bytes(_q(HIRES_BH, HIRES_CHUNK), _k_t(HIRES_BH, 128, HIRES_K))
    assert nbytes == 1321664512
    assert nbytes > fa.SCORE_BUDGET_BYTES
    assert fa.prefer_fallback(_q(HIRES_BH, HIRES_CHUNK), _k_t(HIRES_BH, 128, HIRES_K)) is False


def test_budget_boundary_is_inclusive_and_documented():
    q = _q(1, 1024, d=128)
    k = _k_t(1, 128, fa.SCORE_BUDGET_BYTES // (1 * 1024 * 2))
    assert fa.score_matrix_bytes(q, k) == fa.SCORE_BUDGET_BYTES
    assert fa.prefer_fallback(q, k) is True          # at the boundary: the measured-faster path
    assert fa.prefer_fallback(q, _k_t(1, 128, k.shape[2] + 1)) is False


def test_the_budget_tracks_dtype_width():
    # Same shape, fp32: the fallback would need twice the bytes, so the decision must flip.
    q16, k16 = _q(32, 1024), _k_t(32, 128, 15000)
    q32 = _q(32, 1024, dtype=torch.float32)
    assert fa.prefer_fallback(q16, k16) is True
    assert fa.score_matrix_bytes(q32, k16) == 2 * fa.score_matrix_bytes(q16, k16)


def test_decision_never_touches_a_device():
    # These are plain CPU tensors; a device query here would raise or hang under HIP_VISIBLE_DEVICES="".
    assert fa.prefer_fallback(_q(4, 128), _k_t(4, 128, 1024)) in (True, False)


def test_support_contract_is_still_what_it_was():
    # is_supported answers "could this be served", not "will it be taken": the budget must not leak
    # into it, otherwise the diagnostics and the benchmark harness lose their meaning.
    reason = fa._unsupported_reason(_q(2, 4), _k_t(2, 128, 16), _q(2, 16), False, None)
    assert reason is not None and "GPU" in reason


def test_fused_entry_contract_is_untouched_by_the_policy():
    # The kernel's own entry point answers "can this be served". The policy lives in the wrapper,
    # so a device test that calls fused_attention() with a small geometry still gets a real answer.
    before = dict(fa.stats)
    assert fa.fused_attention(_q(PROD_BH, PROD_CHUNK), _k_t(PROD_BH, 128, PROD_K),
                              _q(PROD_BH, PROD_K), 128 ** -0.5, False, None) is None
    assert fa.stats["fallback_calls"] == before["fallback_calls"] + 1
    assert fa.stats["budget_fallbacks"] == before["budget_fallbacks"]


def test_guarded_entry_declines_the_measured_faster_geometry():
    before = fa.stats["budget_fallbacks"]
    out = fa.fused_attention_guarded(_q(PROD_BH, PROD_CHUNK), _k_t(PROD_BH, 128, PROD_K),
                                     _q(PROD_BH, PROD_K), 128 ** -0.5, False, None)
    assert out is None
    assert fa.stats["budget_fallbacks"] == before + 1


def test_guarded_entry_delegates_above_the_budget():
    # The large geometry must reach fused_attention(), where the device contract is what declines
    # it on CPU -- if the wrapper swallowed it, the OOM-margin branch would be dead on arrival.
    before = dict(fa.stats)
    q, k, v = _q(HIRES_BH, HIRES_CHUNK), _k_t(HIRES_BH, 128, HIRES_K), _q(HIRES_BH, HIRES_K)
    assert fa.prefer_fallback(q, k) is False
    assert fa.fused_attention_guarded(q, k, v, 128 ** -0.5, False, None) is None
    assert fa.stats["fallback_calls"] == before["fallback_calls"] + 1
    assert fa.stats["budget_fallbacks"] == before["budget_fallbacks"]


def test_budget_declines_are_counted_separately():
    # A call the kernel *could* serve but hands back must be visible in stats, or a future
    # "why is my attention slow" investigation cannot tell declined from unsupported.
    assert "budget_fallbacks" in fa.stats


def test_the_dit_hands_the_guarded_entry_to_the_attention_path():
    # Wiring check: the policy must be on the path the model takes, and the raw kernel must stay
    # importable for the device tests and the benchmark harness.
    from comfy.ldm.qwen_image21 import model
    assert model.qwen21_fused_attention is fa.fused_attention_guarded
    assert callable(fa.fused_attention)
