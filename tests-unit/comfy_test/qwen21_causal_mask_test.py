"""Regression test for the Qwen-Image-2.1 text-segment causal mask contract.

Defect: build_sequence() emitted a 2-D torch.bool block-causal mask and handed it to
optimized_attention. ComfyUI's generic backends consume the mask with `attn_scores +=
mask`, so True promoted to +1.0 and False to +0.0: allowed positions got a logit bias and
forbidden positions were not suppressed at all. On attention_sub_quad / attention_split
(the backends a GPU without PyTorch attention resolves to) the mask was effectively absent;
attention_basic raised instead, because its bool branch rearranges a mask it assumes is 1-D.

Design rules, learned the hard way while writing the probe this replaces:
  * ONE deterministic Q/K/V draw shared by the reference and every implementation. An
    earlier comparison built the reference from different random tensors and reported a
    meaningless ~0.13 "error".
  * The reference is an independent float64 masked softmax, not a second code path.
  * Causality is asserted structurally (a row must not change when a key it may not see is
    perturbed), not only by aggregate error.
  * Relative error is peak-normalised. Attention outputs legitimately pass through zero, so
    an elementwise |a-r|/|r| reports O(10) even for a perfect fp16 result.
  * The old implementation is reproduced as a *data* choice (the bool tensor), so the
    negative control does not depend on reverting code.

CPU only: it never initialises CUDA, so it runs without a device.
"""

import ast
import inspect

import pytest
import torch

from comfy.ldm.modules import sub_quadratic_attention as sqa
from comfy.ldm.modules.attention import attention_basic, attention_split, attention_sub_quad

try:
    from comfy.ldm.qwen_image21.model import causal_mask
except ImportError:  # the pre-fix tree has no such helper
    causal_mask = None

# The axes the production DiT uses (32 heads, head_dim 128), reduced to keep the run cheap
# while preserving everything the mask logic depends on.
B, H, D = 1, 4, 8

# A missing helper must be reported once, by test_causal_mask_helper_exists, with the message
# that names the defect. Everything below it needs the helper to exist, so it skips rather
# than failing with "NoneType is not callable" - which would make the negative control
# indistinguishable from a genuinely wrong mask implementation.
missing_helper = pytest.mark.skipif(causal_mask is None, reason="no causal_mask() on this tree")


def test_causal_mask_helper_exists():
    assert causal_mask is not None, (
        "comfy.ldm.qwen_image21.model has no causal_mask(): the text segment still carries "
        "a bool mask, which sub_quad/split consume as a +1.0 bias rather than a mask"
    )


@missing_helper
def test_emitted_mask_is_additive():
    m = causal_mask(6, 0, torch.device("cpu"), torch.float16)
    assert m.dtype.is_floating_point, f"mask must be additive-capable, got {m.dtype}"
    assert tuple(m.shape) == (6, 6)
    allowed = torch.ones(6, 6, dtype=torch.bool).tril(0)
    assert bool((m[allowed] == 0).all()), "allowed positions must be exactly 0"
    assert bool(torch.isneginf(m[~allowed]).all()), "forbidden positions must be -inf"
    assert bool((m.diagonal() == 0).all()), "every token must attend to itself"
    strict_upper = torch.triu(torch.ones(6, 6, dtype=torch.bool), 1)
    assert bool(torch.isneginf(m[strict_upper]).all())


@missing_helper
@pytest.mark.parametrize("n,length", [(6, 0), (6, 3), (8, 2), (10, 10)])
def test_matches_ones_tril_contract(n, length):
    """The mask must equal the intended keep-set exactly, including shifted prefixes."""
    m = causal_mask(n, length, torch.device("cpu"), torch.float16)
    assert tuple(m.shape) == (n, length + n)
    allowed = torch.ones(n, length + n, dtype=torch.bool).tril(length)
    assert bool(((m == 0) == allowed).all())


def shared_qkv(dtype, n, nk, b=B, h=H, d=D, seed=None):
    g = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 20260928 + n)
    q = torch.randn(b * h, n, d, generator=g, dtype=torch.float64)
    k = torch.randn(b * h, nk, d, generator=g, dtype=torch.float64)
    v = torch.randn(b * h, nk, d, generator=g, dtype=torch.float64)
    return q.to(dtype=dtype), k.to(dtype=dtype), v.to(dtype=dtype)


def reference_attention(q, k, v, mask_add, d):
    scores = (q.double() @ k.double().transpose(-1, -2)) * (d ** -0.5)
    scores = scores + mask_add.double()
    return torch.softmax(scores, dim=-1) @ v.double()


def run_sub_quad(q, k, v, mask, qcs, kvc):
    return sqa.efficient_dot_product_attention(
        query=q, key_t=k.transpose(-1, -2).contiguous(), value=v,
        query_chunk_size=qcs, kv_chunk_size=kvc, kv_chunk_size_min=None,
        use_checkpoint=False, upcast_attention=False, mask=mask)


def max_abs(a, r):
    return (a.double() - r).abs().max().item()


def peak_rel(a, r):
    scale = r.abs().max().item()
    return 0.0 if scale == 0.0 else max_abs(a, r) / scale


def to_backend_layout(t4, tokens):
    """(B*H, tokens, D) -> (B, tokens, H*D): what optimized_attention passes in."""
    return t4.view(B, H, tokens, D).permute(0, 2, 1, 3).reshape(B, tokens, H * D)


def from_backend_layout(out, tokens):
    """(B, tokens, H*D) -> (B*H, tokens, D), so it can meet the reference."""
    return out.view(B, tokens, H, D).permute(0, 2, 1, 3).reshape(B * H, tokens, D)


@missing_helper
@pytest.mark.parametrize("n,nk,length", [(6, 6, 0), (7, 10, 3), (9, 9, 0)])
def test_matches_fp64_reference_both_chunk_paths(n, nk, length):
    q, k, v = shared_qkv(torch.float64, n, nk, seed=99 + n)
    add = causal_mask(n, length, torch.device("cpu"), torch.float64)
    ref = reference_attention(q, k, v, add, D)

    single = run_sub_quad(q, k, v, add, 4096, nk)               # one kv chunk: default path
    two_level = run_sub_quad(q, k, v, add, 2, max(1, nk // 3))  # streaming combine

    assert max_abs(single, ref) < 1e-10
    assert max_abs(two_level, ref) < 1e-10
    assert max_abs(single, two_level) < 1e-10
    assert torch.isfinite(single).all() and torch.isfinite(two_level).all()


@missing_helper
def test_forbidden_key_cannot_change_another_rows_output():
    """Structural causality: a key a row may not see must not move that row's output."""
    n = 8
    q, k, v = shared_qkv(torch.float64, n, n, seed=4242)
    add = causal_mask(n, 0, torch.device("cpu"), torch.float64)
    base = run_sub_quad(q, k, v, add, 4096, n)

    k2, v2 = k.clone(), v.clone()
    k2[:, n - 1, :] += 1e3   # visible only to the last row
    v2[:, n - 1, :] += 1e3
    pert = run_sub_quad(q, k2, v2, add, 4096, n)

    assert max_abs(pert[:, : n - 1, :], base[:, : n - 1, :]) < 1e-12
    assert max_abs(pert[:, n - 1, :], base[:, n - 1, :]) > 1e-6


@missing_helper
def test_bool_representation_is_still_wrong_on_the_same_backends():
    """Guards the fix: the historical bool tensor must remain measurably broken here.

    This pins the backends' *contract*, not this model's code: if the generic paths ever
    learn to treat a bool mask as a mask, this fails and says so in its message.
    """
    n = 6
    q, k, v = shared_qkv(torch.float64, n, n, seed=31337)
    add = causal_mask(n, 0, torch.device("cpu"), torch.float64)
    bool_mask = torch.ones(n, n, dtype=torch.bool).tril(0)
    ref = reference_attention(q, k, v, add, D)

    old = run_sub_quad(q, k, v, bool_mask, 4096, n)
    new = run_sub_quad(q, k, v, add, 4096, n)
    assert max_abs(old, ref) > 1e-2, (
        "the bool form no longer behaves as a +1.0 bias here; the backends changed contract, "
        "and that deserves an explicit decision rather than an accident")
    assert max_abs(new, ref) < 1e-10
    assert max_abs(old, ref) / max(max_abs(new, ref), 1e-12) > 1000


@missing_helper
@pytest.mark.parametrize("backend", [attention_sub_quad, attention_split])
def test_production_backends_agree_with_reference_on_additive_mask(backend):
    """The two backends that add the mask to the scores, called the way the DiT calls them."""
    n, nk = 64, 64
    q, k, v = shared_qkv(torch.float64, n, nk, seed=777)
    add = causal_mask(n, 0, torch.device("cpu"), torch.float64)
    ref = reference_attention(q, k, v, add, D)

    out = backend(to_backend_layout(q, n), to_backend_layout(k, nk), to_backend_layout(v, nk),
                  H, mask=add, attn_precision=torch.float64)
    assert max_abs(from_backend_layout(out, n), ref) < 1e-3


@missing_helper
def test_attention_basic_accepts_the_additive_mask():
    """attention_basic raised on the bool 2-D mask; the additive form must work."""
    n = 16
    q, k, v = shared_qkv(torch.float64, n, n, seed=808)
    add = causal_mask(n, 0, torch.device("cpu"), torch.float64)
    ref = reference_attention(q, k, v, add, D)

    out = attention_basic(to_backend_layout(q, n), to_backend_layout(k, n), to_backend_layout(v, n),
                          H, mask=add, attn_precision=torch.float64)
    assert max_abs(from_backend_layout(out, n), ref) < 1e-3


def test_attention_basic_raises_on_the_historical_bool_mask():
    """The other half of the defect: attention_basic's bool branch assumes a 1-D key-padding
    mask, so coercing the dtype there would not have fixed this call site either."""
    n = 16
    q, k, v = shared_qkv(torch.float32, n, n, seed=808)
    bool_mask = torch.ones(n, n, dtype=torch.bool).tril(0)
    with pytest.raises(Exception):
        attention_basic(to_backend_layout(q, n), to_backend_layout(k, n), to_backend_layout(v, n),
                        H, mask=bool_mask, attn_precision=torch.float32)


@missing_helper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_production_dtype_stays_accurate(dtype):
    n = 512
    q, k, v = shared_qkv(dtype, n, n, b=1, h=8, d=64, seed=555)
    add = causal_mask(n, 0, torch.device("cpu"), dtype)
    ref = reference_attention(q.float(), k.float(), v.float(), add.float(), 64)
    # The bound is derived from the dtype, not hand-picked: bf16 carries ~8x less mantissa
    # than fp16, so one magic constant would either pass bf16 meaninglessly or fail it
    # spuriously.
    tol = 8 * torch.finfo(dtype).eps
    for qcs, kvc, label in [(4096, n, "single-chunk"), (128, 128, "two-level")]:
        out = run_sub_quad(q, k, v, add, qcs, kvc)
        assert torch.isfinite(out).all(), label
        assert max_abs(out, ref) < tol, f"{label}: {max_abs(out, ref)} vs tol {tol}"
        assert peak_rel(out, ref) < tol, f"{label}: peak-rel {peak_rel(out, ref)} vs tol {tol}"


@pytest.mark.parametrize("kvc", [1, 2, 4])
def test_fully_masked_kv_chunk_does_not_corrupt_the_combine(kvc):
    """A dead (all -inf) key chunk must not poison the two-level running softmax."""
    n, nk = 8, 16
    q, k, v = shared_qkv(torch.float64, n, nk, seed=606)
    add = torch.zeros(n, nk, dtype=torch.float64)
    add.masked_fill_(~torch.ones(n, nk, dtype=torch.bool).tril(nk - n), float("-inf"))
    ref = reference_attention(q, k, v, add, D)
    out = run_sub_quad(q, k, v, add, 2, kvc)
    assert max_abs(out, ref) < 1e-10
    assert torch.isfinite(out).all()


def test_build_sequence_no_longer_constructs_a_bool_mask_inline():
    """Structural guard on the real call site, not a string match on one line.

    build_sequence() is where the segments are made and the defect lived exactly in the mask
    expression there. Walking its AST for any `ones(..., dtype=torch.bool, ...)` catches a
    revert to the old form even when written differently; requiring a causal_mask() call
    catches the subtler regression where the helper survives but the call site stops using it.
    """
    model_module = __import__("comfy.ldm.qwen_image21.model",
                              fromlist=["QwenImage21Transformer2DModel"])
    cls = model_module.QwenImage21Transformer2DModel
    class_node = next(n for n in ast.walk(ast.parse(inspect.getsource(inspect.getmodule(cls))))
                      if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    fn = next(m for m in class_node.body
              if isinstance(m, ast.FunctionDef) and m.name == "build_sequence")

    bool_ones = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "ones"
        and any(kw.arg == "dtype" and "bool" in ast.dump(kw.value) for kw in node.keywords)
    ]
    assert not bool_ones, (
        f"build_sequence() still builds a bool mask at line(s) {[n.lineno for n in bool_ones]}; "
        "sub_quad/split add it to the scores as +1.0 instead of masking"
    )
    helper_calls = [node for node in ast.walk(fn)
                    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "causal_mask"]
    assert helper_calls, "build_sequence() must construct the text-segment mask through causal_mask()"
