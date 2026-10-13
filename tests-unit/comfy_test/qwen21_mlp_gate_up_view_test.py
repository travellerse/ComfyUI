"""Contract tests for the Qwen-Image-2.1 `gate_up` non-materialising transposed view
(Q21-OPT-035).

What this file is responsible for, and what it deliberately is not:

  * It pins the *interface* of `linear_transposed` after the `materialise` parameter was
    added: the default must keep every pre-existing caller's behaviour, because the only
    other caller in the tree is the down projection inside `linear_input_act`.
  * It pins the two properties that make the optimisation safe rather than merely fast:
    the strided product reaches `swiglu` without being materialised anywhere first, and
    the block's residual stream is still contiguous (the down projection still materialises).
  * It pins the interrupt-checkpoint count. Output equality cannot detect a lost
    `run_every_op()`, so this counts calls and includes a negative control that proves the
    assertion can actually fail.

It does not measure performance: the end-to-end gain is measured evidence in
evidence/reports/47_*.md and must not be re-derived from a benchmark here.

CPU only: it never runs a device op. comfy.model_management resolves the torch device at
import, so cpu mode is declared first, exactly as the sibling tests do.
"""

import inspect
import pathlib

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as ops  # noqa: E402
from comfy.ldm.qwen_image21.model import SwiGLUFeedForward  # noqa: E402
import comfy.ldm.qwen_image21.model as q21  # noqa: E402

OPS = ops.disable_weight_init
# Same relationships as the real module (mlp_ratio=3, gate_up maps dim -> 2*hidden).
DIM, HIDDEN = 32, 96
# The two row counts the production edit workflow really produces: 5120 target tokens per
# sampling step, 5167 for the once-per-run prefix-fill pass (47 text + 5120 reference).
PRODUCTION_ROWS = (5120, 5167)


def _mlp(dim=DIM, hidden=HIDDEN, seed=0):
    """A fused SwiGLU block with real weights.

    `disable_weight_init.Linear` leaves the weight as allocated memory, so an uninitialised
    module is a NaN source rather than a test subject.
    """
    mlp = SwiGLUFeedForward(dim, hidden, fused=True, operations=OPS)
    for lin, fan in ((mlp.gate_up, dim), (mlp.out, hidden)):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / fan ** 0.5)
    return mlp


def _count_run_every_op(fn, *a, **kw):
    n = {"calls": 0}
    real = ops.run_every_op

    def spy():
        n["calls"] += 1
        return real()

    ops.run_every_op = spy
    try:
        with torch.no_grad():
            out = fn(*a, **kw)
    finally:
        ops.run_every_op = real
    return n["calls"], out


# --------------------------------------------------------------- A. default behaviour


def test_materialise_defaults_to_the_shipped_behaviour():
    assert inspect.signature(ops.linear_transposed).parameters["materialise"].default is True


def test_default_call_still_materialises_and_the_opt_out_does_not():
    mlp = _mlp(seed=1)
    act = torch.randn(1, 24, HIDDEN) / HIDDEN ** 0.5
    with torch.no_grad():
        default = ops.linear_transposed(mlp.out, act)
        explicit = ops.linear_transposed(mlp.out, act, materialise=True)
        view = ops.linear_transposed(mlp.out, act, materialise=False)
    assert default.is_contiguous() and explicit.is_contiguous()
    assert view.is_contiguous() is False
    assert torch.equal(default, explicit) and torch.equal(default, view)
    # the exact layout the optimisation relies on: token axis becomes stride 1 and the
    # feature axis strides over the token count, i.e. a transpose, not a copy
    n, d = view.shape[1], view.shape[2]
    assert view.stride() == (d * n, 1, n), view.stride()
    assert default.stride() == (n * d, d, 1), default.stride()
    assert default.shape == view.shape and default.dtype == view.dtype


def test_bias_is_still_applied_after_the_view():
    lin = OPS.Linear(HIDDEN, DIM, bias=True)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(DIM, HIDDEN, generator=g) / HIDDEN ** 0.5)
        lin.bias.copy_(torch.randn(DIM, generator=g))
        act = torch.randn(1, 16, HIDDEN, generator=g)
        view = ops.linear_transposed(lin, act, materialise=False)
        copy = ops.linear_transposed(lin, act, materialise=True)
        ref = torch.nn.functional.linear(act, lin.weight, lin.bias)
    assert torch.equal(view, copy)
    tol = 16 * torch.finfo(torch.float32).eps * float(ref.abs().max())
    assert float((view - ref).abs().max()) < tol


def test_the_view_branch_still_uses_the_layers_own_cast_bias_weight_once():
    """A GGUF layer dequantises only from inside its own cast_bias_weight; a path that
    fetched `linear.weight` directly would silently skip that lifecycle."""
    calls = []
    lin = OPS.Linear(DIM, 2 * HIDDEN, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(2 * HIDDEN, DIM) / DIM ** 0.5)

    def spy_cast(input=None, **kw):
        calls.append(1)
        return lin.weight, None

    lin.cast_bias_weight = spy_cast
    with torch.no_grad():
        out = ops.linear_transposed(lin, torch.randn(1, 12, DIM), materialise=False)
    assert len(calls) == 1
    assert out.shape == (1, 12, 2 * HIDDEN)


# ------------------------------------------------------------------ B. the gate_up path


def test_production_form_has_no_experiment_switch():
    """The A/B/A scaffold's `_gate_up_view` must not have survived into production."""
    assert not hasattr(q21, "_gate_up_view")
    assert not hasattr(q21, "_gate_up_view_enabled")


def test_gate_up_product_reaches_swiglu_as_a_view():
    seen = {}
    real = ops.linear_input_act

    def spy(linear, x, *a, **kw):
        seen["contiguous"] = x.is_contiguous()
        seen["shape"] = tuple(x.shape)
        return real(linear, x, *a, **kw)

    mlp = _mlp(seed=3)
    ops.linear_input_act = spy
    try:
        with torch.no_grad():
            mlp(torch.randn(1, 16, DIM))
    finally:
        ops.linear_input_act = real
    assert seen["contiguous"] is False, "gate_up's product must not be materialised"
    assert seen["shape"] == (1, 16, 2 * HIDDEN)


def test_non_materialising_call_happens_exactly_once_per_forward():
    """Exactly one `materialise=False` call per block forward: the down projection must
    keep materialising, otherwise the residual contract below stops holding."""
    seen = {"non_materialising": 0, "materialising": 0}
    real = ops.linear_transposed

    def spy(linear, x, materialise=True):
        seen["non_materialising" if not materialise else "materialising"] += 1
        return real(linear, x, materialise=materialise)

    mlp = _mlp(seed=4)
    ops.linear_transposed = spy
    try:
        with torch.no_grad():
            mlp(torch.randn(1, 16, DIM))
    finally:
        ops.linear_transposed = real
    assert seen == {"non_materialising": 1, "materialising": 1}, seen


def test_block_output_stays_contiguous_for_the_residual_contract():
    """`_gated_residual` writes in place into x; the MLP result must stay contiguous so
    the cost of the view never gets transferred to the residual stream."""
    mlp = _mlp(seed=5)
    with torch.no_grad():
        out = mlp(torch.randn(1, 16, DIM))
    assert out.is_contiguous() is True
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("rows", PRODUCTION_ROWS)
def test_production_shapes_match_the_materialised_reference_within_tolerance(rows):
    """The optimisation changes the rounding path, not the accuracy class: compare against
    an fp64 reference rather than against the old code, and keep the bound dtype-derived."""
    mlp = _mlp(seed=6)
    x = torch.randn(1, rows, DIM) / DIM ** 0.5
    with torch.no_grad():
        got = mlp(x)
        sub = slice(0, 32)
        gu = x[:, sub].double() @ mlp.gate_up.weight.double().T
        act = ops._eager_input_act(gu, "swiglu", None, 0.0)
        ref = act @ mlp.out.weight.double().T
    err = float(got[:, sub].double().sub(ref).abs().max())
    tol = 64 * torch.finfo(torch.float32).eps * float(ref.abs().max())
    assert got.shape == (1, rows, DIM)
    assert torch.isfinite(got).all()
    assert err < tol, (rows, err, tol)


# ------------------------------------------------------- C. the run_every_op contract


def test_gate_up_path_keeps_its_interrupt_checkpoint():
    """Two checkpoints per fused MLP forward: one for gate_up (explicit, since the
    transposed helper does not call it) and one inside linear_input_act."""
    mlp = _mlp(seed=7)
    n, _ = _count_run_every_op(mlp, torch.randn(1, 16, DIM))
    assert n == 2, n


def test_the_checkpoint_assertion_can_actually_fail():
    """Negative control: a variant that drops the explicit run_every_op() must be caught.
    Without this, the assertion above could be vacuously true."""
    mlp = _mlp(seed=8)
    real = q21.SwiGLUFeedForward.forward

    def without_checkpoint(self, x):
        return ops.linear_input_act(
            self.out,
            ops.linear_transposed(self.gate_up, x, materialise=False),
            "swiglu", transposed=self.transposed_down_projection)

    q21.SwiGLUFeedForward.forward = without_checkpoint
    try:
        n, _ = _count_run_every_op(mlp, torch.randn(1, 16, DIM))
    finally:
        q21.SwiGLUFeedForward.forward = real
    assert n == 1, (n, "the counter is not sensitive to a missing checkpoint")


# --------------------------------------------------------------------- D. scope guard


def test_transposed_form_is_reached_from_exactly_two_sites():
    """One definition, one pre-existing caller (the down projection) and one new caller
    (gate_up). Widening this must be a deliberate change to this test."""
    root = pathlib.Path(ops.__file__).parent
    sites = []
    for py in sorted(root.rglob("*.py")):
        try:
            text = py.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "linear_transposed(" in line and not line.lstrip().startswith("#"):
                sites.append((str(py.relative_to(root)), lineno, "def " in line))
    in_ops = [s for s in sites if s[0] == "ops.py"]
    in_q21 = [s for s in sites if s[0].endswith("qwen_image21/model.py")]
    elsewhere = [s for s in sites if s not in in_ops and s not in in_q21]
    assert len(in_ops) == 2 and sum(1 for s in in_ops if s[2]) == 1, in_ops
    assert len(in_q21) == 1 and not in_q21[0][2], in_q21
    assert elsewhere == [], elsewhere


def test_the_similarly_named_mmdit_module_is_untouched():
    """`comfy/ldm/modules/diffusionmodules/mmdit.py` has its own SwiGLUFeedForward.
    A name-based grep could easily drag it into scope; it must not gain the flag."""
    import comfy.ldm.modules.diffusionmodules.mmdit as mmdit
    assert not hasattr(mmdit, "_gate_up_view")
    src = pathlib.Path(mmdit.__file__).read_text()
    assert "linear_transposed" not in src
    assert "materialise" not in src


def test_sibling_qwen_and_flux_models_have_not_grown_the_flag():
    import comfy.ldm.flux.model as flux_model
    import comfy.ldm.qwen_image.model as qwen_image_model
    for mod in (flux_model, qwen_image_model):
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                assert not hasattr(obj, "transposed_down_projection"), "%s.%s" % (
                    mod.__name__, name)
