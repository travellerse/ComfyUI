"""Dispatch and correctness tests for the Qwen-Image-2.1 Triton fused attention.

The dispatch contract is the important half of this file. The fused kernel exists to
replace one specific shape of work, so the tests pin both directions: the geometry it
is meant to take, and every geometry that must be handed back to the existing
implementation. A silent downgrade is fine; a silent wrong answer is not, so the
programming-bug path is asserted to raise rather than to fall back.
"""

import pytest
import torch

from comfy.ldm.qwen_image21 import fused_attention as fa
from comfy.ldm.modules import sub_quadratic_attention as sqa


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS = 32
HEAD_DIM = 128
QUERY_CHUNK = 1024


def _geometry(n_tokens, heads=HEADS, head_dim=HEAD_DIM, dtype=torch.float16, device="cuda"):
    """The tensors as `_get_attention_scores_no_kv_chunking` receives them."""
    q = torch.randn(heads, QUERY_CHUNK, head_dim, dtype=dtype, device=device)
    k = torch.randn(heads, n_tokens, head_dim, dtype=dtype, device=device)
    v = torch.randn(heads, n_tokens, head_dim, dtype=dtype, device=device)
    return q, k.transpose(-1, -2).contiguous(), v


@pytest.fixture(autouse=True)
def _reset_stats():
    before = dict(fa.stats)
    yield
    fa.stats.update(before)
    fa._logged_unsupported.clear()


def test_supported_geometry_takes_the_fused_path():
    q, key_t, v = _geometry(3072)
    with torch.no_grad(), torch.inference_mode():
        assert fa.is_supported(q, key_t, v) is True
        out = fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None)
    assert out is not None
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
    assert fa.stats["fused_calls"] == 1
    assert fa.stats["fallback_calls"] == 0


@pytest.mark.parametrize("mask_kind,upcast,reason", [
    ("present", False, "mask"),
    ("absent", True, "upcast"),
])
def test_mask_and_upcast_fall_back(mask_kind, upcast, reason):
    q, key_t, v = _geometry(3072)
    mask = torch.zeros(HEADS, QUERY_CHUNK, 3072, dtype=torch.float16, device="cuda") if mask_kind == "present" else None
    with torch.no_grad(), torch.inference_mode():
        out = fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, upcast, mask)
    assert out is None, reason
    assert fa.stats["fallback_calls"] == 1


def test_head_dim_falls_back():
    # head_dim is a constexpr, so an unvalidated value must not be compiled on the fly.
    q, key_t, v = _geometry(3072, head_dim=64)
    with torch.no_grad(), torch.inference_mode():
        assert fa.is_supported(q, key_t, v) is False
        assert fa.fused_attention(q, key_t, v, 64 ** -0.5, False, None) is None


def test_dtype_falls_back():
    q, key_t, v = _geometry(3072, dtype=torch.bfloat16)
    with torch.no_grad(), torch.inference_mode():
        assert fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None) is None
    assert fa.stats["fallback_calls"] == 1


def test_cpu_tensors_fall_back():
    q, key_t, v = _geometry(3072, dtype=torch.float16, device="cpu")
    with torch.no_grad(), torch.inference_mode():
        assert fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None) is None


def test_mismatched_head_dim_falls_back():
    q, key_t, _ = _geometry(3072)
    v = torch.randn(HEADS, 3072, 64, dtype=torch.float16, device="cuda")
    with torch.no_grad(), torch.inference_mode():
        assert fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None) is None


def test_grad_enabled_falls_back():
    # out= softmax and the kernel both lack autograd support, so an enabled graph must
    # keep the original path rather than produce a tensor that cannot be differentiated.
    q, key_t, v = _geometry(3072)
    with torch.enable_grad():
        assert torch.is_grad_enabled()
        assert fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None) is None


def test_numerics_match_fp32_reference_at_qwen_shapes():
    for n_tokens in (3072, 6144):
        q, key_t, v = _geometry(n_tokens)
        with torch.no_grad(), torch.inference_mode():
            out = fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None)
            ref = torch.softmax((q.float() @ key_t.float()) * HEAD_DIM ** -0.5, dim=-1) @ v.float()
        assert out is not None, n_tokens
        diff = (out.float() - ref).abs()
        assert diff.max().item() < 5e-4, (n_tokens, diff.max().item())
        assert diff.mean().item() < 1e-4, (n_tokens, diff.mean().item())
        assert torch.isfinite(out).all()


def test_padding_tail_does_not_leak_probability():
    # k_tokens deliberately not a multiple of BLOCK_N, so the final tile is padded.
    # A zero-filled padding column would otherwise enter the denominator as real mass.
    n_tokens = 1024 * 6 + 37
    q, key_t, v = _geometry(n_tokens)
    with torch.no_grad(), torch.inference_mode():
        out = fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None)
        ref = torch.softmax((q.float() @ key_t.float()) * HEAD_DIM ** -0.5, dim=-1) @ v.float()
    assert out is not None
    assert ((out.float() - ref).abs().max().item()) < 5e-4


class _FakeTriton:
    """Enough of the triton module for the failure paths, and nothing more."""

    def __init__(self, kernel, error):
        self.cdiv = lambda a, b: -(-a // b)
        self.runtime = type("runtime", (), {"errors": type("errors", (), {"OutOfResources": error})})
        self._kernel = kernel

    def __getitem__(self, grid):  # used as the module by the test via the tuple below
        return self


def _install_fake_triton(kernel_call, error=type("OutOfResources", (Exception,), {})):
    triton_like = _FakeTriton(None, error)
    fa._triton_state = ((triton_like, kernel_call), None)


def test_internal_failure_is_not_swallowed():
    """A defect must surface. Only an out-of-resource hardware limit may fall back."""
    q, key_t, v = _geometry(3072)
    sentinel = RuntimeError("shape bug")

    class Boom:
        def __getitem__(self, grid):
            def call(*a, **k):
                raise sentinel
            return call

    _install_fake_triton(Boom())
    try:
        with torch.no_grad(), torch.inference_mode(), pytest.raises(RuntimeError) as exc:
            fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None)
        assert exc.value is sentinel
    finally:
        fa._triton_state = None


def test_out_of_resources_falls_back():
    """A kernel that does not fit the device is a limit, not a defect."""
    q, key_t, v = _geometry(3072)
    error = type("OutOfResources", (Exception,), {})

    class TooBig:
        def __getitem__(self, grid):
            def call(*a, **k):
                raise error("out of resource: shared memory")
            return call

    _install_fake_triton(TooBig(), error=error)
    try:
        with torch.no_grad(), torch.inference_mode():
            assert fa.fused_attention(q, key_t, v, HEAD_DIM ** -0.5, False, None) is None
    finally:
        fa._triton_state = None
    assert fa.stats["compile_failures"] == 1


def test_sub_quad_uses_inplace_softmax_and_forwards_fused():
    """The shared attention path must honour both new inputs."""
    q, key_t, v = _geometry(3072)
    scale = HEAD_DIM ** -0.5
    with torch.no_grad(), torch.inference_mode():
        baseline = sqa._get_attention_scores_no_kv_chunking(q, key_t, v, scale, False, None)
        inplace = sqa._get_attention_scores_no_kv_chunking(q, key_t, v, scale, False, None, inplace_softmax=True)
    assert torch.allclose(baseline, inplace, atol=0, rtol=0), "in-place softmax must be bit-identical here"

    seen = {}

    def declines(query, key_t, value, fused_scale, upcast, mask):
        seen["called"] = True
        return None

    with torch.no_grad(), torch.inference_mode():
        fell_back = sqa._get_attention_scores_no_kv_chunking(q, key_t, v, scale, False, None, fused_attention=declines)
    assert seen.get("called") is True
    assert torch.allclose(baseline, fell_back, atol=0, rtol=0)


def test_sub_quad_softmax_stays_out_of_place_under_grad():
    q, key_t, v = _geometry(3072)
    scale = HEAD_DIM ** -0.5
    with torch.no_grad():
        baseline = sqa._get_attention_scores_no_kv_chunking(q, key_t, v, scale, False, None)
    with torch.enable_grad():
        # Must not raise: the out= form is illegal under autograd.
        gradfull = sqa._get_attention_scores_no_kv_chunking(q, key_t, v, scale, False, None, inplace_softmax=True)
    assert torch.allclose(baseline, gradfull, atol=0, rtol=0)


def test_shared_path_defaults_are_unchanged():
    """Other models call the same function without the new arguments.

    Both additions are opt-in, so every existing caller must keep the exact previous
    behaviour: out-of-place softmax and no fused hook.
    """
    import inspect
    sig = inspect.signature(sqa._get_attention_scores_no_kv_chunking)
    assert sig.parameters["inplace_softmax"].default is False
    assert sig.parameters["fused_attention"].default is None

    from comfy.ldm.modules.attention import attention_sub_quad
    sig2 = inspect.signature(attention_sub_quad.__wrapped__)
    assert sig2.parameters["inplace_softmax"].default is False
    assert sig2.parameters["fused_attention"].default is None


def test_global_attention_backend_selection_is_untouched():
    """The Qwen optimisation must not change what any other model is dispatched to."""
    import comfy.ldm.modules.attention as attn
    from comfy import model_management
    assert attn.optimized_attention is attn.attention_sub_quad
    assert model_management.sage_attention_enabled() is False
    assert attn.optimized_attention_masked is attn.optimized_attention

