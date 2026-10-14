"""Contract tests for the query chunk size the Qwen-Image-2.1 DiT pins at its two
attention call sites (Q21-OPT-039).

What this file is responsible for:

  * `block_causal_attention` and `prefix_cached_attention` in
    `comfy/ldm/qwen_image21/model.py` each make exactly one `optimized_attention` call,
    and each passes `query_chunk_size=1024` explicitly. That 1024 is a measured decision
    (evidence/OPTIMIZATION_LEDGER.md Q21-OPT-029, reports 42/50): the free-VRAM chooser in
    `attention_sub_quad` picked 4096 then 2048 across the steps of one run, so the chunk has
    to be a function of the shapes and not of how much memory happens to be free.
  * The value must not be silently inherited from the callee. `attention_sub_quad`'s own
    default is `None` and its body falls back to **512**, which is a different number:
    dropping the keyword would not raise, it would quietly change the chunk.
  * The two sites must not drift apart. They are the cached and the fill branch of the same
    budget, so one at 1024 and one at anything else is a bug even if both "work".

What this file deliberately does NOT claim:

  * It does not claim every ComfyUI model should use 1024, or that 1024 is right for a
    geometry other than the one measured. Scope is the two call sites in this module.
  * It does not forbid legitimate overrides. `attention.py` rebinds the module global
    `optimized_attention` by backend at import time, and `wrap_attn` honours
    `transformer_options["optimized_attention_override"]`; `FixedChunkAttentionPolicy`
    (attention.py:1003) is a separate mechanism the DiT path does not use. None of those are
    touched here. A hook node replacing attention wholesale is a different, legitimate
    contract and pinning it would be wrong.
  * It measures nothing. There is no timing assertion in this file, because the performance
    evidence is measured and lives in the reports; re-deriving it here would be a second,
    weaker copy of the same claim.

Why this file reads source text instead of importing modules -- and why that is the right
tool for this particular contract:

  The sibling tests in this directory import `comfy.ops`, which imports `comfy.float`, which
  imports `comfy_kitchen`; that chain opens `/dev/kfd` and `/dev/dri/renderD128` and leaves
  `torch.cuda.is_initialized() == True` (measured in Phase 55; see
  `/tmp/q21-r55/probe_hip_context.py`). Phase 55 is a CPU-only phase that must not
  initialise a device context, and this contract is *purely syntactic*: it is about which
  keyword argument a call site carries. An AST test expresses exactly that, needs no device,
  and cannot pass for the wrong reason (no dispatch, no monkeypatching, no fixture state).
  Behavioural testing would be the wrong tool here: the value is a literal at the call site,
  so there is no runtime behaviour to observe that the source text does not already state.

  The trade-off recorded honestly: an AST test is sensitive to refactors that keep the
  behaviour (e.g. hoisting the kwargs into a shared dict). Test 6 below is the guard for
  that -- it fails with a message naming the refactor as the cause rather than reporting a
  bare "keyword missing", so a future edit gets a diagnostic, not a mystery.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "comfy" / "ldm" / "qwen_image21" / "model.py"
ATTENTION = REPO_ROOT / "comfy" / "ldm" / "modules" / "attention.py"

PINNED_CHUNK = 1024
CALLEE_BODY_FALLBACK = 512
CALLEE_SITES = ("block_causal_attention", "prefix_cached_attention")


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree, name):
    """The function named `name`, searching nested scopes so the inner `attn` closures are
    reachable. Ambiguity is an error rather than a silent first match: two functions with the
    same name would make "which call site" unanswerable."""
    found = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        pytest.fail(f"expected exactly one function named {name!r}, found {len(found)} "
                    f"at lines {[f.lineno for f in found]} -- the contract target is ambiguous")
    return found[0]


def _optimized_attention_calls(func):
    """Every `optimized_attention(...)` call inside `func`, including nested defs.

    Matches on the bare name only: the call site uses the module-level `optimized_attention`
    global, and if a refactor made it an attribute call that is a real change of contract and
    should be reported as one, not silently skipped."""
    return [n for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "optimized_attention"]


def _keyword_value(call, kw):
    for k in call.keywords:
        if k.arg != kw:
            continue
        if isinstance(k.value, ast.Constant):
            return k.value.value, "literal"
        return None, "non-constant"
    return None, "absent"


def _pinned_chunk(call):
    return _keyword_value(call, "query_chunk_size")


# ---------------------------------------------------------------- the real call sites


def test_model_file_and_functions_are_where_the_contract_lives():
    """A moving-target guard first: if these files move, the rest of this file would be
    asserting about nothing."""
    assert MODEL.is_file(), f"{MODEL} is not a file -- this test has lost its target"
    assert ATTENTION.is_file(), f"{ATTENTION} is not a file -- this test has lost its target"
    tree = _tree(MODEL)
    for name in CALLEE_SITES:
        assert _function(tree, name) is not None


@pytest.mark.parametrize("site", CALLEE_SITES)
def test_each_attention_site_pins_the_query_chunk_size(site):
    calls = _optimized_attention_calls(_function(_tree(MODEL), site))
    assert len(calls) == 1, (
        f"{site} makes {len(calls)} optimized_attention calls; this test knows how to grade "
        f"exactly one, so a new site must be added to the contract deliberately")
    value, how = _pinned_chunk(calls[0])
    assert how == "literal", (
        f"{site}: query_chunk_size is {how}; the pinned value must be a literal here, "
        f"otherwise the budget is no longer a function of the shapes. If this was an "
        f"intentional refactor, update this test as part of that change, not to make it pass")
    assert value == PINNED_CHUNK, (
        f"{site}: query_chunk_size={value!r}, expected {PINNED_CHUNK} "
        f"(Q21-OPT-029; see evidence/OPTIMIZATION_LEDGER.md)")


@pytest.mark.parametrize("site", CALLEE_SITES)
def test_each_site_also_pins_inplace_softmax(site):
    """The paired half of the same memory argument: `inplace_softmax` removes the second
    full-size score tensor the softmax would otherwise write. Chunk size alone bounds one
    transient; dropping this silently doubles it while every chunk-size test still passes."""
    calls = _optimized_attention_calls(_function(_tree(MODEL), site))
    value, how = _keyword_value(calls[0], "inplace_softmax")
    assert how == "literal" and value is True, f"{site}: inplace_softmax is {how}={value!r}"


def test_the_two_sites_cannot_drift_apart():
    """Both branches bound the same score matrix, so they must carry the same number."""
    tree = _tree(MODEL)
    values = {}
    for site in CALLEE_SITES:
        calls = _optimized_attention_calls(_function(tree, site))
        values[site], _ = _pinned_chunk(calls[0])
    assert len(set(values.values())) == 1, f"the two sites disagree: {values}"
    assert values[CALLEE_SITES[0]] == PINNED_CHUNK


def test_no_third_unpinned_site_silently_appears_in_this_module():
    """The whole DiT path has exactly two `optimized_attention` call sites. A third one
    without the keyword would take the callee's 512 fallback and no per-site test would
    notice, because there would be nothing telling on it."""
    calls = [n for n in ast.walk(_tree(MODEL))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "optimized_attention"]
    assert len(calls) == len(CALLEE_SITES), (
        f"found {len(calls)} optimized_attention call sites in model.py but the contract "
        f"covers {len(CALLEE_SITES)}; grade the new site explicitly")


# ------------------------------------------------------------- the callee cannot supply it


def test_callee_default_is_none_so_the_keyword_carries_the_whole_value():
    """`attention_sub_quad`'s signature default must not be able to produce 1024 by itself.

    If this ever becomes a literal 1024, the call sites' keyword stops being the thing that
    pins the budget, and this file's premise changes -- which is exactly what should force a
    revisit rather than a quiet pass."""
    fn = _function(_tree(ATTENTION), "attention_sub_quad")
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                        fn.args.defaults))
    node = defaults.get("query_chunk_size")
    assert node is not None, "attention_sub_quad no longer defaults query_chunk_size"
    assert isinstance(node, ast.Constant) and node.value is None, (
        f"attention_sub_quad(query_chunk_size=...) default is now {node!r}; the DiT "
        f"contract assumed the caller is the only source of 1024")


def test_callee_fallback_is_512_not_1024_so_omission_would_change_behaviour():
    """The safety of the *mutation check* below rests on this: dropping the keyword must be
    a real behaviour change, not a no-op. Asserted from the callee's own body."""
    src = _function(_tree(ATTENTION), "attention_sub_quad")
    assigned = [n.value.value for n in ast.walk(src)
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, int)
                and any(isinstance(t, ast.Name) and t.id == "query_chunk_size"
                        for t in n.targets)]
    assert CALLEE_BODY_FALLBACK in assigned, (
        f"attention_sub_quad no longer falls back to {CALLEE_BODY_FALLBACK} "
        f"(found {assigned}); re-derive what omission would do before trusting this file")


# ----------------------------------------------------------------- sensitivity, by design
#
# §4.3 requires proof that the assertions can fail. These build synthetic source text in
# memory -- the production files are never written to -- and run the *same* detectors over
# them, so what is under test is the detector, not a copy of the fixture.


def _variant(site, replacement):
    """The real model.py with one of `site`'s two call lines rewritten in text form.

    Deliberately a textual splice of the single call, then re-parsed: it proves the test
    looks at that call, and it never touches the file on disk."""
    lines = MODEL.read_text().splitlines(keepends=True)
    target = None
    tree = ast.parse("".join(lines))
    for c in _optimized_attention_calls(_function(tree, site)):
        target = c
    assert target is not None
    end = target.end_lineno
    blob = "".join(lines[target.lineno - 1:end])
    patched = replacement(blob)
    assert patched != blob, "the mutation did not change the source -- not a real variant"
    return "".join(lines[:target.lineno - 1]) + patched + "".join(lines[end:])


def _observed_chunk_literal(site):
    """Whatever the graded source currently carries, so the three checks below test the
    DETECTOR rather than re-asserting the contract value. If they hard-coded 1024 they would
    fail for the wrong reason (baseline drift) and stop distinguishing 'the detector is blind'
    from 'someone changed the number'."""
    calls = _optimized_attention_calls(_function(_tree(MODEL), site))
    value, how = _pinned_chunk(calls[0])
    assert how == "literal", f"{site}: cannot build a variant of a non-literal"
    return value


def test_removing_the_keyword_would_fail_the_contract():
    here = _observed_chunk_literal("block_causal_attention")
    src = _variant("block_causal_attention",
                   lambda b: b.replace(f"query_chunk_size={here}, ", ""))
    calls = _optimized_attention_calls(_function(ast.parse(src), "block_causal_attention"))
    value, how = _pinned_chunk(calls[0])
    assert (value, how) == (None, "absent"), (
        "the detector did not notice a deleted keyword -- the contract is not binding")


def test_replacing_the_value_would_fail_the_contract():
    here = _observed_chunk_literal("prefix_cached_attention")
    other = 512 if here != 512 else 256
    src = _variant("prefix_cached_attention",
                   lambda b: b.replace(f"query_chunk_size={here}", f"query_chunk_size={other}"))
    calls = _optimized_attention_calls(_function(ast.parse(src), "prefix_cached_attention"))
    value, _ = _pinned_chunk(calls[0])
    assert value == other, "the detector did not read the value it was given"


def test_a_hoisted_kwargs_dict_is_reported_as_a_missing_keyword_not_a_silence():
    """The known AST fragility, tested instead of ignored: `**kw` produces a `keyword` with
    `arg is None`, which the lookup above reads as 'absent'. It must fail *loudly*, and it
    must not be confused with a legitimate override mechanism."""
    here = _observed_chunk_literal("block_causal_attention")
    src = _variant("block_causal_attention",
                   lambda b: b.replace(f"query_chunk_size={here}, inplace_softmax=True, ",
                                       "**chunk_kwargs, "))
    calls = _optimized_attention_calls(_function(ast.parse(src), "block_causal_attention"))
    value, how = _pinned_chunk(calls[0])
    assert how == "absent"
    # and the co-pinned softmax flag goes the same way, so the pair fails together
    assert _keyword_value(calls[0], "inplace_softmax")[1] == "absent"


def test_production_source_is_untouched_by_this_file():
    """Guard for the mutation tests above: they must have left the real file alone."""
    digest_now = MODEL.read_bytes()
    assert b"query_chunk_size=1024" in digest_now
    assert digest_now.count(b"query_chunk_size=") == len(CALLEE_SITES), (
        "model.py no longer has exactly one pinned chunk per graded site")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
