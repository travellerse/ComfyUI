"""Contract tests for the Qwen-Image-2.1 MLP down-projection's transposed GEMM.

Why a test rather than a benchmark: the change swaps which operand rocBLAS sees, so the two
forms are *not* bit-identical, and the only thing worth pinning is the contract around that -
accuracy must not get worse, nothing may become NaN/Inf, the shared helper's default must keep
every existing caller byte-for-byte unchanged, and the shape guard must not fire on a config it
was not measured on.

CPU only: it never runs a device op. comfy.model_management resolves the torch device at import
(model_management.py:364), so cpu mode is declared first, exactly as the sibling tests do.
"""

import inspect

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as ops  # noqa: E402
from comfy.ldm.qwen_image21.model import SwiGLUFeedForward  # noqa: E402

# Small, but with the same shape relationships as the real module: the down projection maps
# hidden -> dim, and the guard fires when hidden > 2 * dim.
DIM, HIDDEN, TOKENS = 64, 192, 48
OPS = ops.disable_weight_init


def _down(dtype=torch.float32, seed=0):
    """The down projection: Linear(hidden, dim, bias=False).

    Built through the ops class the model actually uses, not torch.nn.Linear, because
    comfy.ops.cast_bias_weight expects a CastWeightBiasOp (it reads weight_function/bias_function).
    """
    g = torch.Generator().manual_seed(seed)
    lin = OPS.Linear(HIDDEN, DIM, bias=False, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(DIM, HIDDEN, generator=g, dtype=dtype))
    return lin


def _activated(dtype=torch.float32, seed=1, tokens=TOKENS):
    """The input the down projection actually sees: swiglu(gate_up(x)) -> (tokens, hidden)."""
    g = torch.Generator().manual_seed(seed)
    pre = torch.randn(tokens, 2 * HIDDEN, generator=g, dtype=dtype)
    return ops._eager_input_act(pre, "swiglu", None, 0.0)


def test_shared_helper_default_keeps_existing_callers_unchanged():
    """A new optional kwarg must not move any current call site."""
    assert inspect.signature(ops.linear_input_act).parameters["transposed"].default is False


def test_default_path_is_bit_identical_to_the_previous_behaviour():
    """transposed=False must leave linear_input_act exactly as it was: a plain linear(act(x))."""
    lin = _down()
    x = torch.randn(TOKENS, 2 * HIDDEN, generator=torch.Generator().manual_seed(2))
    got = ops.linear_input_act(lin, x, "swiglu")
    want = lin(ops._eager_input_act(x, "swiglu", None, 0.0))
    assert torch.equal(got, want)


def test_transposed_matches_the_fp64_reference():
    lin = _down()
    act = _activated()
    ref64 = act.double() @ lin.weight.double().T
    plain = lin(act)
    transposed = ops.linear_transposed(lin, act)

    # tolerance derived from the dtype and the magnitude, not hand-picked: one constant either
    # passes meaninglessly or fails spuriously depending on the draw
    tol = 8 * torch.finfo(torch.float32).eps * float(ref64.abs().max())
    for name, out in (("plain", plain), ("transposed", transposed)):
        assert torch.isfinite(out.detach()).all(), name
        assert float((out.detach().double() - ref64).abs().max()) < tol, (name, tol)


def test_transposed_is_not_materially_less_accurate():
    """The change buys speed; it must not pay for it with accuracy."""
    lin = _down(seed=5)
    act = _activated(seed=6)
    ref64 = act.double() @ lin.weight.double().T
    e_plain = float((lin(act).detach().double() - ref64).abs().mean())
    e_trans = float((ops.linear_transposed(lin, act).detach().double() - ref64).abs().mean())
    assert e_trans <= 2 * e_plain + 1e-9, (e_plain, e_trans)


@pytest.mark.parametrize("batch", [1, 3])
def test_transposed_accepts_the_real_rank(batch):
    """The DiT calls the MLP with (batch, tokens, hidden). The first implementation used .t()
    and only passed the 2-D micro-benchmark shape; this pins the rank the model actually uses."""
    lin = _down(seed=11)
    g = torch.Generator().manual_seed(12)
    pre = torch.randn(batch, TOKENS, 2 * HIDDEN, generator=g)
    act = ops._eager_input_act(pre, "swiglu", None, 0.0)
    assert act.dim() == 3

    plain = lin(act)
    transposed = ops.linear_transposed(lin, act)
    assert transposed.shape == plain.shape == (batch, TOKENS, DIM)
    ref64 = act.detach().double() @ lin.weight.detach().double().T
    tol = 8 * torch.finfo(torch.float32).eps * float(ref64.abs().max())
    assert float((transposed.detach().double() - ref64).abs().max()) < tol


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transposed_stays_accurate_in_half_precision(dtype):
    lin = _down(dtype=dtype, seed=7)
    act = _activated(dtype=dtype, seed=8)
    ref64 = act.double() @ lin.weight.double().T
    plain = lin(act)
    transposed = ops.linear_transposed(lin, act)

    tol = 8 * torch.finfo(dtype).eps * float(ref64.abs().max())
    assert torch.isfinite(transposed.detach()).all()
    assert float((transposed.detach().double() - ref64).abs().max()) < tol
    assert float((transposed.detach().double() - ref64).abs().mean()) <= \
        2 * float((plain.detach().double() - ref64).abs().mean()) + 1e-9


def test_shape_guard_fires_only_on_the_measured_relationship():
    """The real module is dim=4096 / hidden=12288; the guard must not fire on equal widths."""
    assert SwiGLUFeedForward(4096, 12288, fused=True, operations=OPS).transposed_down_projection is True
    assert SwiGLUFeedForward(DIM, HIDDEN, fused=True, operations=OPS).transposed_down_projection is True
    assert SwiGLUFeedForward(64, 64, fused=True, operations=OPS).transposed_down_projection is False
    assert SwiGLUFeedForward(64, 96, fused=True, operations=OPS).transposed_down_projection is False
    assert SwiGLUFeedForward(64, 128, fused=True, operations=OPS).transposed_down_projection is False
    assert SwiGLUFeedForward(64, 129, fused=True, operations=OPS).transposed_down_projection is True


def test_guard_is_scoped_to_this_module_only():
    """The flag is set in SwiGLUFeedForward.__init__; the sibling models must not have grown it."""
    import comfy.ldm.flux.model as flux_model
    import comfy.ldm.qwen_image.model as qwen_image_model

    for mod in (flux_model, qwen_image_model):
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                assert not hasattr(obj, "transposed_down_projection"), "%s.%s" % (mod.__name__, name)
