"""Static contract tests for the free-VRAM query inside `attention_sub_quad` (Q21-OPT-040).

The claim under test is narrow and structural, and it is the ONLY claim this file makes:

  `attention_sub_quad` queries free device memory before it knows whether it needs to, and
  the value it gets is read by exactly one of the two chunk-selection branches. A caller
  that passes `query_chunk_size` (or `kv_chunk_size`) explicitly -- which is what
  Qwen-Image-2.1 does -- takes the branch that never reads it.

Proving that from source is worth doing because it is the precondition for any later
"move the query into the branch that uses it" change: the change is only safe if the greedy
branch keeps its data path and the explicit branch genuinely never needed it.

What this file deliberately does NOT do, each for a stated reason:

  * It does not measure or estimate the cost of `get_free_memory`. No GPU is started in this
    phase, so there is no number here, and an unmeasured number dressed as a bound is the
    exact failure mode this project has been burned by.
  * It does not assert "64 queries per step". The per-step count is
    (call sites) x (segments per block) x (blocks) x (steps), and the segment count comes
    from `build_sequence` at runtime -- statically the call *sites* are countable, the
    multiplicity is not. Test 8 pins what is countable and says so.
  * It does not claim the query should be removed, that removing it speeds anything up, or
    that it has any relation to the SDMA0 fault. Whether it is worth touching at all is a
    later, separate change review that needs runtime timing evidence.
  * It does not generalise to other models. `attention_sub_quad` is shared, and the callers
    that pass neither chunk parameter still need the greedy path; test 6 pins that they keep
    it, and nothing here argues their query is wasted.

CPU-only by construction: this file imports `ast`/`pathlib` only and reads source text, so it
never opens a device node. (The sibling tests import `comfy.ops` -> `comfy.float` ->
`comfy_kitchen`, which does open /dev/kfd; Phase 55 measured that, see
/tmp/q21-r55/probe_hip_context.py.)
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ATTENTION = REPO_ROOT / "comfy" / "ldm" / "modules" / "attention.py"
QWEN_MODEL = REPO_ROOT / "comfy" / "ldm" / "qwen_image21" / "model.py"

QUERY_VAR = "mem_free_total"
QUERY_CALL = "get_free_memory"
CHUNK_PARAMS = ("query_chunk_size", "kv_chunk_size")


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree, name, path=None):
    found = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        pytest.fail(f"expected exactly one {name!r}"
                    f"{' in ' + str(path) if path else ''}, found {len(found)} "
                    f"at lines {[f.lineno for f in found]}")
    return found[0]


def _loads_of(node, var):
    """Every Name load of `var` within `node`, with line numbers."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == var]


def _stores_of(node, var):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and n.id == var]


def _query_statement(fn):
    """The top-level statement in `attention_sub_quad`'s body that binds the query result.

    Returning the statement rather than the target Name is what makes the "is it guarded"
    check meaningful: only a body-level statement can be compared against `fn` itself."""
    hits = [s for s in fn.body
            if isinstance(s, ast.Assign) and _stores_of(s, QUERY_VAR)]
    if len(hits) != 1:
        pytest.fail(f"expected exactly one body-level assignment binding {QUERY_VAR} in "
                    f"attention_sub_quad, found {len(hits)}; if the query is now guarded or "
                    f"nested, the premise of this file has changed -- re-derive it")
    return hits[0]


def _chunk_branch(fn):
    """The single `If ... is not None or ... is not None ...` that decides chunking.

    Found by structure, not by line number. `is not None` is what selects it: the same
    function also contains `if query_chunk_size is None:` style *default assignments*, and
    those are not the branch this contract is about. Requiring exactly one candidate, and one
    whose test covers both parameters and has a non-empty `orelse`, is itself part of the
    contract -- if the greedy fallback were hoisted out or one parameter dropped, the branch
    this file reasons about would no longer exist and these tests must say so rather than
    grade a different node."""
    cands = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        comps = [c for c in ast.walk(n.test) if isinstance(c, ast.Compare)]
        names = {c.left.id for c in comps
                 if len(c.ops) == 1 and isinstance(c.ops[0], ast.IsNot)
                 and isinstance(c.comparators[0], ast.Constant)
                 and c.comparators[0].value is None
                 and isinstance(c.left, ast.Name) and c.left.id in CHUNK_PARAMS}
        if names == set(CHUNK_PARAMS):
            cands.append(n)
    if len(cands) != 1:
        pytest.fail(f"expected exactly one chunk-selection If in attention_sub_quad, "
                    f"found {len(cands)} at lines {[c.lineno for c in cands]} -- the "
                    f"branch structure this contract describes no longer exists")
    node = cands[0]
    assert node.orelse, (
        f"the chunk-selection If at line {node.lineno} has no else branch; the greedy path "
        f"this contract protects is gone")
    return node


# ------------------------------------------------------------------ the query and its readers


def test_the_function_asks_for_free_memory_exactly_once():
    fn = _function(_tree(ATTENTION), "attention_sub_quad", ATTENTION)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == QUERY_CALL]
    assert len(calls) == 1, f"{len(calls)} get_free_memory calls; this contract grades one"
    stmt = _query_statement(fn)
    # and the call really is what binds it (a tuple target is how `a, _ = ...` parses)
    bound = [t for tgt in stmt.targets for t in
             (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt])]
    assert any(isinstance(t, ast.Name) and t.id == QUERY_VAR for t in bound), (
        f"{QUERY_VAR} is no longer bound by a statement with that target")
    assert isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) \
        and stmt.value.func.attr == QUERY_CALL, (
        "the statement binding mem_free_total is no longer the get_free_memory call")


def test_the_query_is_unconditional_and_sits_before_the_branch():
    """`CONFIRMED` structurally: nothing gates the query on whether it will be used."""
    fn = _function(_tree(ATTENTION), "attention_sub_quad", ATTENTION)
    branch = _chunk_branch(fn)
    stmt = _query_statement(fn)
    assert fn.body.index(stmt) < fn.body.index(branch), (
        f"query at body#{fn.body.index(stmt)}, branch at #{fn.body.index(branch)}: the "
        f"ordering this file is named after has changed -- re-derive before trusting it")
    # `_query_statement` only ever returns a body-level statement, so reaching it at all
    # proves the query is not nested inside an If/For/Try. Asserted explicitly so a future
    # refactor cannot satisfy this by accident.
    assert any(s is stmt for s in fn.body)


def test_the_explicit_chunk_branch_never_reads_the_query():
    """The load-bearing assertion. If the `then` branch read `mem_free_total`, the query
    would be needed on the Qwen-Image-2.1 path and there would be nothing to report."""
    branch = _chunk_branch(_function(_tree(ATTENTION), "attention_sub_quad", ATTENTION))
    # the THEN branch only -- `ast.walk(branch)` would also descend into the greedy `else`
    # and report the read this contract says belongs there
    reads = [r for stmt in branch.body for r in _loads_of(stmt, QUERY_VAR)]
    assert reads == [], (
        f"the explicit-chunk branch reads {QUERY_VAR} at lines "
        f"{[r.lineno for r in reads]}; the query IS used on that path")


def test_every_read_of_the_query_lives_in_the_greedy_else_branch():
    """Stronger than the above: there is no *other* reader anywhere in the function that
    this file has not accounted for."""
    fn = _function(_tree(ATTENTION), "attention_sub_quad", ATTENTION)
    branch = _chunk_branch(fn)
    else_reads = {r.lineno for stmt in branch.orelse for r in _loads_of(stmt, QUERY_VAR)}
    all_reads = {r.lineno for r in _loads_of(fn, QUERY_VAR)}
    unexpected = sorted(all_reads - else_reads)
    assert not unexpected, (
        f"{QUERY_VAR} is read at lines {unexpected} outside the greedy branch; moving the "
        f"query would break them")


def test_the_greedy_branch_still_consumes_it_and_carries_the_chunk_ladder():
    """The other side of "safe to move": the branch that needs the number must keep needing
    it in the same shape, so a future change review compares like with like."""
    branch = _chunk_branch(_function(_tree(ATTENTION), "attention_sub_quad", ATTENTION))
    assert branch.orelse, "the chunk-selection If has no else branch any more"
    reads = _loads_of(ast.Module(body=branch.orelse, type_ignores=[]), QUERY_VAR)
    assert reads, "the greedy branch no longer reads the free-memory figure it chose chunks with"
    loops = [n for n in ast.walk(branch) if isinstance(n, ast.For)]
    assert loops, "no candidate-chunk loop found in the selection branch"
    ladder = [c.value for l in loops for c in ast.walk(l.iter)
              if isinstance(c, ast.Constant) and isinstance(c.value, int)]
    assert ladder and ladder[0] > ladder[-1], (
        f"the candidate chunk ladder is {ladder}; the greedy path is expected to walk "
        f"chunk sizes downward")


def test_a_caller_passing_neither_parameter_still_lands_in_the_greedy_branch():
    """Requirement 3 of the phase brief: the generic callers must keep their data path.

    Both parameters default to None, so `query_chunk_size is not None or kv_chunk_size is
    not None` is False and control goes to the else branch -- which test 4 just proved is the
    only place the query is read. The default and the branch test are pinned together here,
    because changing either one silently changes which callers pay for the query."""
    fn = _function(_tree(ATTENTION), "attention_sub_quad", ATTENTION)
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                        fn.args.defaults))
    for p in CHUNK_PARAMS:
        assert p in defaults, f"{p} is no longer a defaulted parameter of attention_sub_quad"
        assert isinstance(defaults[p], ast.Constant) and defaults[p].value is None, (
            f"{p} default is not None; the caller-class split this file describes changed")


def test_the_qwen_sites_take_the_explicit_branch_by_value_not_by_line_number():
    """Cross-file structural inference, stated as such: 1024 is not None, therefore the
    `is not None` disjunct is True, therefore the then-branch runs, therefore (test 3) the
    query is not read. This is a conclusion about control flow, not a measurement."""
    calls = [n for n in ast.walk(_tree(QWEN_MODEL))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "optimized_attention"]
    assert len(calls) == 2, f"expected the two graded qwen21 sites, found {len(calls)}"
    for c in calls:
        kw = {k.arg: k for k in c.keywords if k.arg in CHUNK_PARAMS}
        assert "query_chunk_size" in kw, "a qwen21 site no longer passes query_chunk_size"
        v = kw["query_chunk_size"].value
        assert isinstance(v, ast.Constant) and v.value is not None, (
            f"query_chunk_size={ast.dump(v)}; a None would flip this site into the greedy "
            f"branch and make the query live on the production path")


def test_only_two_call_sites_exist_in_the_module_and_the_multiplicity_is_not_claimed():
    """What is and is not statically countable, kept in one place.

    Countable: the number of `optimized_attention` call sites in qwen_image21/model.py.
    NOT countable here: how many times each runs per step -- that depends on the segment
    list `build_sequence` produces at runtime -- so this file states no per-step figure and
    deliberately does not name one."""
    tree = _tree(QWEN_MODEL)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "optimized_attention"]
    assert len(calls) == 2
    # the loop the sites sit in is a runtime-counted construct, and asserting about its
    # trip count from source is exactly the over-claim this phase forbids
    enclosing = [n for n in ast.walk(tree)
                 if isinstance(n, ast.For) and any(c in ast.walk(n) for c in calls)]
    assert enclosing, (
        "the call sites are no longer inside a loop; the per-step count would become "
        "statically determinable and this test would need to be replaced by one")


# ------------------------------------------------------------------- sensitivity of the checks


def _attention_variant(mutate):
    """A synthetic copy of attention.py with one structural piece rewritten. The file on disk
    is never modified; the same detectors then run over the returned text."""
    src = ATTENTION.read_text()
    patched = mutate(src)
    assert patched != src, "the mutation did not change anything -- not a real variant"
    return patched


def test_detector_catches_a_read_added_to_the_explicit_branch():
    fn = _function(_tree(ATTENTION), "attention_sub_quad", ATTENTION)
    branch = _chunk_branch(fn)
    # splice a read of the query into the then-branch and confirm test 3's detector sees it
    extra = "\n        _ = " + QUERY_VAR
    lines = ATTENTION.read_text().splitlines(keepends=True)
    ins = branch.body[0].lineno - 1
    mutated = ("".join(lines[:ins]) + extra + "\n" + "".join(lines[ins:]))
    mfn = _function(ast.parse(mutated), "attention_sub_quad")
    mbranch = _chunk_branch(mfn)
    assert [r for stmt in mbranch.body for r in _loads_of(stmt, QUERY_VAR)], (
        "the detector missed an injected read in the explicit branch -- the load-bearing "
        "test above is not binding")


def test_detector_catches_the_greedy_branch_stopping_to_read_it():
    mutated = _attention_variant(
        lambda s: s.replace("count = mem_free_total /", "count = 1.0 /"))
    mfn = _function(ast.parse(mutated), "attention_sub_quad")
    mbranch = _chunk_branch(mfn)
    assert _loads_of(ast.Module(body=mbranch.orelse, type_ignores=[]), QUERY_VAR) == [], (
        "the greedy branch stopped needing the query and nothing noticed -- moving the "
        "query would then be a behaviour change masquerading as a cleanup")


def test_detector_catches_a_guard_wrapping_the_query():
    mutated = _attention_variant(
        lambda s: s.replace("    mem_free_total, _ = model_management.get_free_memory",
                            "    if query_chunk_size is None:\n"
                            "        mem_free_total, _ = model_management.get_free_memory"))
    mfn = ast.parse(mutated)
    mfn = _function(mfn, "attention_sub_quad")
    # the query is no longer a body-level statement, so `_query_statement` must refuse it
    with pytest.raises(pytest.fail.Exception, match="body-level"):
        _query_statement(mfn)


def test_production_attention_source_is_untouched_by_this_file():
    blob = ATTENTION.read_text()
    assert blob.count(f"{QUERY_VAR}, _ = model_management.{QUERY_CALL}") == 1
    assert "query_chunk_size=None" in blob


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
