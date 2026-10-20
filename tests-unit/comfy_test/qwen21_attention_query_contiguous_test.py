"""Structural contract for Q21-OPT-068: the 3-D path of `attention_sub_quad` always feeds
`baddbmm` a contiguous query, whether or not the key repack is enabled.

The claim is narrow: `query = query.contiguous()` must not be nested inside the
`if repacked is not None:` body. The repack flag (`Q21_TRITON_REPACK_KEY`) is off by
default, so a contiguous query that only happens on the repack path is a contiguous query
production never sees -- which is exactly the state this file exists to prevent from
coming back.

Why structure and not a number: the end-to-end effect (-1.9% step, -5.9% attention,
1024x1280, seed 5152) is measured in evidence/reports/64 §18, and it depends on which
Tensile tile rocBLAS picks, which no CPU test can observe. What a CPU test *can* pin is
the shape of the code that decides the operand layout.

CPU-only: reads source text with `ast`, never opens a device node.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ATTENTION = REPO_ROOT / "comfy" / "ldm" / "modules" / "attention.py"


def _function(name="attention_sub_quad"):
    tree = ast.parse(ATTENTION.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {ATTENTION}")


def _is_query_contiguous(stmt):
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    if not (isinstance(target, ast.Name) and target.id == "query"):
        return False
    value = stmt.value
    return (isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "contiguous")


def _repack_if(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "repacked" in ast.dump(node.test):
            return node
    return None


def _ids(nodes):
    return {id(n) for n in nodes}


def test_contiguous_query_exists_once():
    hits = [n for n in ast.walk(_function()) if _is_query_contiguous(n)]
    assert len(hits) == 1, f"expected exactly one 'query = query.contiguous()', found {len(hits)}"


def test_contiguous_query_is_not_inside_the_repack_branch():
    fn = _function()
    repack = _repack_if(fn)
    assert repack is not None, "the repack branch disappeared; this test needs updating"
    assert repack.orelse == [], "repack branch grew an else; re-check where contiguous belongs"
    hit = next(n for n in ast.walk(fn) if _is_query_contiguous(n))
    assert id(hit) not in _ids(ast.walk(repack)), (
        "query.contiguous() is nested under 'if repacked is not None', so it only runs when "
        "Q21_TRITON_REPACK_KEY is on -- which is off by default (Q21-OPT-068)")


def test_repack_branch_still_installs_the_packed_key():
    repack = _repack_if(_function())
    assigns_key = [n for n in ast.walk(repack)
                   if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == "key"]
    assert assigns_key, "the repack branch no longer assigns key; Q21-OPT-066 regressed"


def test_contiguous_query_runs_before_the_contiguous_key_option():
    """`contiguous_key` is a separate, still-off option (Q21-OPT-066). The unconditional
    query copy must happen before it so the two never get confused in review."""
    fn = _function()
    body = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "contiguous_key" in ast.dump(node.test):
            body = node
            break
    assert body is not None, "contiguous_key option disappeared"
    hit = next(n for n in ast.walk(fn) if _is_query_contiguous(n))
    assert hit.lineno < body.lineno, "query.contiguous() moved below the contiguous_key option"
