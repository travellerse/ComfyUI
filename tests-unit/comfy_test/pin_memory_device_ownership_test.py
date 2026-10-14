"""Phase 56 device-ownership contract tests for pin/unpin lifetime (Q21-OPT-038 candidate).

Companion to `pin_memory_unregister_ordering_test.py` (Phase 55, carried over). That file
grades the *ordering* contract; this one grades the *data model* the candidate introduced to
make the ordering achievable, plus the failure paths the candidate has to get right.

Design under test (see /tmp/q21-r56/opt038_call_graph.md section 6):

    PINNED_MEMORY[ptr] = (size, device)        # device is what the OWNER passed at pin time
    pin_memory(tensor, evict_active=True, device=None)
    unpin_memory(tensor):
        no record            -> warn, return False          (unchanged, never touches driver)
        size mismatch        -> warn, return False          (unchanged, never touches driver)
        device recorded      -> torch.cuda.synchronize(device) BEFORE cudaHostUnregister(ptr)
        synchronize raises   -> warn, return False, record KEPT, driver not touched
        unregister fails     -> warn, return False, record KEPT  (unchanged)
        unregister succeeds  -> pop record, TOTAL -= size, return True

Two things this file is emphatically NOT claiming:

  * It does not show that real ROCm would have faulted. The fake cudart returns what the test
    asks for; `torch.cuda.synchronize` is never called for real -- it would `_lazy_init()` and
    open a device context, which Phase 56 forbids. What is proven is *call order and scope*,
    against the production function body.
  * It does not close OPT-038's causal question. Whether any SDMA0 fault was ever produced by
    an unpin racing a copy stays `UNVERIFIED`; see the SDMA0 boundary test at the end.

`device=None` is a real design case, not a hole being tolerated: `wan/model_animate2.py:189`
pins before it knows a destination GPU (that arrives as a `prefetch(i, device, dtype)`
argument at `:195`), and it already syncs the streams it tracked at `:160`. Turning that into
a guessed device would be worse than leaving it. So device-less registrations must keep
working exactly as before, and the inventory test enumerates them so a new one cannot join
silently.
"""

import ast
import hashlib
import pathlib
import subprocess
import sys

import pytest

import pin_memory_unregister_ordering_test as base  # reuse the harness, do not fork it

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MM = REPO_ROOT / "comfy" / "model_management.py"
MP = REPO_ROOT / "comfy" / "model_patcher.py"
WAN = REPO_ROOT / "comfy" / "ldm" / "wan" / "model_animate2.py"
SEEDVR = REPO_ROOT / "comfy" / "ldm" / "seedvr" / "vae.py"

DEV = ("cuda", 0)        # opaque stand-in; see base._DEV
DEV_OTHER = ("cuda", 1)


# ------------------------------------------------------------------ harness helpers

def _pin():
    return base._load_pin()


def _unpin():
    return base._load_unpin()


def _registered(ns, FakeTensor, ptr=0x7F00_0000_0000, size=4096, device=DEV):
    """Register through the REAL pin_memory, so the ledger shape under test is whatever the
    production code produced -- not something this file wrote by hand."""
    pin, _, events, _, _ = _pin()
    ns_events = events
    ok = pin(FakeTensor(ptr, size), device=device)
    return ok, ns_events


# ---------------------------------------------------------------- 6.1 registration record


def test_registration_stores_pointer_size_and_the_argument_device():
    pin, ns, events, FakeTensor, _ = _pin()
    t = FakeTensor(0x7F11_0000_0000, 40960)
    assert pin(t, device=DEV) is True
    assert ns["PINNED_MEMORY"][0x7F11_0000_0000] == (40960, DEV)
    # the device came from the call, not from a constant baked into the function
    assert pin(FakeTensor(0x7F11_0000_1000, 1024), device=DEV_OTHER) is True
    assert ns["PINNED_MEMORY"][0x7F11_0000_1000] == (1024, DEV_OTHER), (
        "both registrations stored the same device -> the record is not keyed to the argument")


def test_registration_total_stays_a_plain_int():
    """`TOTAL_PINNED_MEMORY` is read arithmetically elsewhere
    (`model_management.py:769`, `model_patcher.py:2118/2143`) and compared numerically by
    `tests-unit/comfy_test/test_seedvr2_internals.py:326/340`. Only the per-pointer *value*
    may change shape."""
    pin, ns, _, FakeTensor, _ = _pin()
    assert pin(FakeTensor(0x7F11_0000_2000, 4096), device=DEV) is True
    assert pin(FakeTensor(0x7F11_0000_3000, 8192), device=DEV) is True
    assert ns["TOTAL_PINNED_MEMORY"] == 4096 + 8192
    assert isinstance(ns["TOTAL_PINNED_MEMORY"], int)


def test_failed_registration_creates_no_phantom_record():
    pin, ns, events, FakeTensor, _ = _pin()
    fn, text = base._source_function(base.MODEL_MANAGEMENT, base.PIN_SRC)
    ns2, Fake2 = base._events_namespace(events, register_ok=False)
    exec(compile(text, str(base.MODEL_MANAGEMENT), "exec"), ns2)
    assert ns2[fn.name](Fake2(0x7F11_0000_4000, 4096), device=DEV) is False
    assert ns2["PINNED_MEMORY"] == {}, "a rejected registration must not be recorded"
    assert ns2["TOTAL_PINNED_MEMORY"] == 0


def test_registration_without_a_device_is_still_possible_and_is_distinguishable():
    """wan/seedvr's status quo must not be silently upgraded into a fake guarantee."""
    pin, ns, _, FakeTensor, _ = _pin()
    assert pin(FakeTensor(0x7F11_0000_5000, 4096)) is True
    assert ns["PINNED_MEMORY"][0x7F11_0000_5000] == (4096, None)


def test_patcher_passes_its_load_device_not_the_enclosing_device_to():
    """The one place where a plausible-looking design would have been wrong.

    `partially_unload(device_to=self.model.offload_device, ...)`
    (`model_management.py:838`) pins at `model_patcher.py:1270`, so threading the enclosing
    `device_to` into the record would store the **CPU** device the weights were evicted to --
    and `torch.cuda.synchronize()` raises on a CPU device. The candidate reads
    `self.load_device` instead; this test pins that choice against the real source."""
    src = MP.read_text()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == "pin_weight_to_device"][0]
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute) and c.func.attr == "pin_memory"]
    assert len(calls) == 1
    kws = {k.arg: ast.dump(k.value) for k in calls[0].keywords}
    assert "device" in kws, (
        "pin_weight_to_device no longer supplies a device; production entries would all be "
        "device-less and the ordering contract would not apply to them")
    assert "load_device" in kws["device"], (
        f"the recorded device is {kws['device']}, expected self.load_device")
    assert "device_to" not in kws["device"], (
        "device_to would be offload_device (CPU) on the partially_unload path")


# ------------------------------------------------------------------- 6.2 happy-path order


def test_event_order_is_sync_then_unregister_then_remove_then_total_update():
    ok, events = None, None
    pin, ns, ev_pin, FakeTensor, _ = _pin()
    assert pin(FakeTensor(0x7F22_0000_0000, 4096), device=DEV) is True
    unpin, ns2, ev_unpin, FakeTensor2, _ = _unpin()
    ns2["PINNED_MEMORY"] = dict(ns["PINNED_MEMORY"])      # the record the real pin produced
    ns2["TOTAL_PINNED_MEMORY"] = ns["TOTAL_PINNED_MEMORY"]
    ev_unpin.clear()
    assert unpin(FakeTensor2(0x7F22_0000_0000, 4096)) is True
    order = [e[0] for e in ev_unpin]
    assert order == ["torch.cuda.synchronize", "cudaHostUnregister"], (
        f"unexpected driver call sequence: {ev_unpin}")
    # the sync targeted the recorded device, by identity, not "current"
    sync_evt = ev_unpin[0]
    assert sync_evt[1] == DEV, f"synchronised {sync_evt[1]!r}, expected {DEV!r}"
    # and the ledger mutations follow the success, in the documented order
    assert 0x7F22_0000_0000 not in ns2["PINNED_MEMORY"]
    assert ns2["TOTAL_PINNED_MEMORY"] == 0


def test_device_less_record_unregisters_without_any_synchronisation():
    """Exactly Phase 55's behaviour, kept on purpose for owners that scope themselves."""
    unpin, ns, events, FakeTensor, _ = _unpin()
    base._seed(ns, 0x7F22_0000_1000, 4096, device=None)
    events.clear()
    assert unpin(FakeTensor(0x7F22_0000_1000, 4096)) is True
    assert [e[0] for e in events] == ["cudaHostUnregister"], events


# ------------------------------------------------------------------- 6.3 failure injection


def test_sync_failure_leaves_the_registration_and_touches_no_driver():
    fn, text = base._source_function(base.MODEL_MANAGEMENT, base.UNPIN_SRC)
    events = []
    ns, FakeTensor = base._events_namespace(events)

    def boom(device=None):
        events.append(("torch.cuda.synchronize", device))
        raise RuntimeError("device busy")

    ns["torch"].cuda.synchronize = boom
    exec(compile(text, str(base.MODEL_MANAGEMENT), "exec"), ns)
    ptr = 0x7F22_0000_2000
    ns["PINNED_MEMORY"][ptr] = (4096, DEV)
    ns["TOTAL_PINNED_MEMORY"] = 4096
    events.clear()
    assert ns["unpin_memory"](FakeTensor(ptr, 4096)) is False
    kinds = [e[0] for e in events]
    assert kinds == ["torch.cuda.synchronize", "logging.warning"], (
        f"after a failed sync the only acceptable calls are the warning; got {events}")
    assert "cudaHostUnregister" not in kinds, "unregistered despite a failed sync"
    assert ns["PINNED_MEMORY"][ptr] == (4096, DEV), "the record was dropped anyway"
    assert ns["TOTAL_PINNED_MEMORY"] == 4096, "the total was decremented for memory still pinned"


def test_no_device_is_guessed_when_the_record_has_none():
    """The fail-closed reading of 'unknown scope': do not synchronise some device on the
    theory that current_device() is probably right."""
    unpin, ns, events, FakeTensor, _ = _unpin()
    ptr = 0x7F22_0000_3000
    base._seed(ns, ptr, 4096, device=None)
    events.clear()
    unpin(FakeTensor(ptr, 4096))
    assert [e[0] for e in events] == ["cudaHostUnregister"]


def test_unknown_pointer_and_size_mismatch_still_never_reach_the_driver():
    unpin, ns, events, FakeTensor, _ = _unpin()
    events.clear()
    assert unpin(FakeTensor(0x7F22_DEAD_0000, 4096)) is False
    assert [e[0] for e in events] == ["logging.warning"]
    events.clear()
    base._seed(ns, 0x7F22_0000_4000, 4096, device=DEV)
    assert unpin(FakeTensor(0x7F22_0000_4000, 9992)) is False
    assert [e[0] for e in events] == ["logging.warning"], (
        "a size mismatch must not synchronise and then fail to unregister either")
    assert ns["PINNED_MEMORY"][0x7F22_0000_4000] == (4096, DEV)


def test_unregister_failure_still_keeps_the_record_after_the_sync():
    fn, text = base._source_function(base.MODEL_MANAGEMENT, base.UNPIN_SRC)
    events = []
    ns, FakeTensor = base._events_namespace(events, register_ok=False)
    exec(compile(text, str(base.MODEL_MANAGEMENT), "exec"), ns)
    ptr = 0x7F22_0000_5000
    ns["PINNED_MEMORY"][ptr] = (4096, DEV)
    ns["TOTAL_PINNED_MEMORY"] = 4096
    events.clear()
    assert ns["unpin_memory"](FakeTensor(ptr, 4096)) is False
    kinds = [e[0] for e in events]
    assert kinds == ["torch.cuda.synchronize", "cudaHostUnregister",
                     "logging.warning", "discard_cuda_async_error"], events
    assert ns["PINNED_MEMORY"][ptr] == (4096, DEV)
    assert ns["TOTAL_PINNED_MEMORY"] == 4096


def test_repeated_release_of_the_same_device_is_safe_and_idempotent():
    """Second unpin finds no record and must warn rather than unregister twice
    (`TOTAL_PINNED_MEMORY` going negative would corrupt `:769`'s arithmetic)."""
    unpin, ns, events, FakeTensor, _ = _unpin()
    ptr = 0x7F22_0000_6000
    base._seed(ns, ptr, 4096, device=DEV)
    ns["TOTAL_PINNED_MEMORY"] = 4096          # as a real pin would have left it
    assert unpin(FakeTensor(ptr, 4096)) is True
    events.clear()
    assert unpin(FakeTensor(ptr, 4096)) is False
    assert [e[0] for e in events] == ["logging.warning"]
    assert ns["TOTAL_PINNED_MEMORY"] == 0


def test_unpin_signature_did_not_move_onto_the_callers():
    """Every existing caller keeps working unedited: the candidate put the scope in the
    ledger, not into a new required argument -- which is why wan/seedvr are unaffected."""
    callers = {}
    for path in (MP, WAN, SEEDVR):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "unpin_memory":
                callers.setdefault(path.name, []).append(len(n.args) + len(n.keywords))
    assert callers, "no unpin_memory call sites found -- the audit has lost its target"
    for f, arities in callers.items():
        assert set(arities) == {1}, f"{f}: unpin_memory now called with {arities} args"


# ------------------------------------------------- 6.3.8 / inventory of device-less owners


def test_device_less_pin_sites_are_an_enumerated_list_not_a_default():
    """A new `pin_memory(...)` call that forgets a device must be *decided*, not inherited.

    Without this, `device=None` quietly blesses every future caller. The list is small and
    each entry has a reason; adding to it is a review event, deleting one is a fix."""
    known = {
        # file: the owner supplies its own scope, so no ledger device is derivable at pin time
        "model_animate2.py": "pins in put(); destination GPU only arrives in prefetch(i, device)",
        "vae.py": "pins in _host_like(); free() already syncs self._device before unpinning",
        # ops.py:249 calls comfy.pinned_memory.pin_memory (a different function that never
        # writes PINNED_MEMORY) and is gated behind aimdo/dynamic-vram, unreachable here.
    }
    found = {}
    for path in sorted((REPO_ROOT / "comfy").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name != "pin_memory":
                continue
            # the core ledger API is model_management.pin_memory, not Tensor.pin_memory()
            recv = ast.unparse(f.value) if isinstance(f, ast.Attribute) and f.value else ""
            if "pin_memory" in recv and "model_management" not in recv and "comfy" not in recv:
                continue
            if not ("model_management" in recv or recv == ""):
                continue
            if any(k.arg == "device" for k in n.keywords):
                continue
            found.setdefault(path.name, []).append(n.lineno)
    assert set(found) <= set(known), (
        f"device-less core pin_memory call sites grew: {sorted(found)} vs known "
        f"{sorted(known)} -- pass a device or justify the owner-managed scope in this list")
    # and the production one really is no longer in the set
    assert "model_patcher.py" not in found, (
        "ModelPatcher pins without a device again: the ordering contract does not apply "
        "to the production weight path")


# --------------------------------------------------------------------- 6.4 sensitivity


def _mm_variant(mutate):
    src = MM.read_text()
    patched = mutate(src)
    assert patched != src, "the mutation did not change anything -- not a real variant"
    return patched


def _run_unpin_on(source_text, seed_device=DEV, sync_boom=None, register_ok=True):
    """Exec ONE variant of `unpin_memory` and return the event log of a release."""
    tree = ast.parse(source_text)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unpin_memory"]
    assert len(fn) == 1
    text = ast.get_source_segment(source_text, fn[0])
    events = []
    ns, FakeTensor = base._events_namespace(events, register_ok=register_ok)
    if sync_boom:
        def boom(device=None):
            events.append(("torch.cuda.synchronize", device))
            raise RuntimeError("boom")
        ns["torch"].cuda.synchronize = boom
    exec(compile(text, "<variant>", "exec"), ns)
    ptr = 0x7F99_0000_0000
    ns["PINNED_MEMORY"][ptr] = (4096, seed_device)
    ns["TOTAL_PINNED_MEMORY"] = 4096
    events.clear()
    ok = ns["unpin_memory"](FakeTensor(ptr, 4096))
    return [e[0] for e in events], ok, ns, events


def test_sensitivity_removing_the_sync_is_detected():
    # neutralised rather than deleted: dropping the only statement in the try would leave
    # `try:` with a comment body and would not even parse
    kinds, ok, _, _ = _run_unpin_on(_mm_variant(
        lambda s: s.replace("            torch.cuda.synchronize(pin_device)", "            pass")))
    assert kinds == ["cudaHostUnregister"], (
        f"the sync was deleted and the release still looked normal: {kinds}")
    assert "torch.cuda.synchronize" not in kinds


def test_sensitivity_wrong_device_is_detected():
    kinds, ok, _, events = _run_unpin_on(MM.read_text(), seed_device=DEV_OTHER)
    synced = [e for e in events if e[0] == "torch.cuda.synchronize"]
    assert synced and synced[0][1] == DEV_OTHER, (
        "the recorded device is not the one synchronised -- a stale record would sync "
        "something else and leave the real copy in flight")


def test_sensitivity_hardcoded_current_device_is_detected():
    """The specific wrong fix: `torch.cuda.synchronize()` with no argument.

    It passes a single-GPU box and silently depends on `current_device()`
    (`torch/cuda/__init__.py synchronize` docstring: "uses the current device ... if device
    is None"). The candidate must read the device off the record."""
    bad = _mm_variant(lambda s: s.replace("torch.cuda.synchronize(pin_device)",
                                          "torch.cuda.synchronize()"))
    kinds, ok, _, events = _run_unpin_on(bad, seed_device=DEV)
    synced = [e for e in events if e[0] == "torch.cuda.synchronize"]
    assert synced and synced[0][1] is None, (
        "variant not applied as expected")
    assert synced[0][1] != DEV, "expected the bad variant to lose the device"


def test_sensitivity_dropping_the_record_before_the_driver_call_is_detected():
    """If `pop` moved above the unregister, a *failing* unregister would report the memory as
    released while the driver still holds the registration -- the ledger would go negative on
    the next real release. Detectable only behaviourally, so it is tested behaviourally."""
    PROD_BLOCK = (
        "    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:\n"
        "        size = PINNED_MEMORY.pop(ptr)[0]\n"
        "        TOTAL_PINNED_MEMORY -= size\n"
        "        return True")
    MOVED_BLOCK = (
        "    size = PINNED_MEMORY.pop(ptr)[0]\n"
        "    TOTAL_PINNED_MEMORY -= size\n"
        "    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:\n"
        "        return True")
    assert PROD_BLOCK in MM.read_text(), "the release block moved on its own; update this probe"
    moved = _mm_variant(lambda s: s.replace(PROD_BLOCK, MOVED_BLOCK))
    fn_tree = ast.parse(moved)
    fn = [n for n in fn_tree.body if isinstance(n, ast.FunctionDef) and n.name == "unpin_memory"]
    assert len(fn) == 1, "the reorder variant is not structurally what was intended"
    events = []
    ns, FakeTensor = base._events_namespace(events, register_ok=False)   # driver refuses
    exec(compile(ast.get_source_segment(moved, fn[0]), "<moved>", "exec"), ns)
    ns["PINNED_MEMORY"][0x7F99_0000_1000] = (4096, DEV)
    ns["TOTAL_PINNED_MEMORY"] = 4096
    assert ns["unpin_memory"](FakeTensor(0x7F99_0000_1000, 4096)) is False
    assert ns["TOTAL_PINNED_MEMORY"] == 0 and not ns["PINNED_MEMORY"], (
        "variant did not exhibit the bug; the detector below proves nothing")
    # production, same injection, must keep the record and the total
    prod_events = []
    ns2, FT2 = base._events_namespace(prod_events, register_ok=False)
    ptree = ast.parse(MM.read_text())
    pfn = [n for n in ptree.body if isinstance(n, ast.FunctionDef) and n.name == "unpin_memory"]
    exec(compile(ast.get_source_segment(MM.read_text(), pfn[0]), "<prod>", "exec"), ns2)
    ns2["PINNED_MEMORY"][0x7F99_0000_1000] = (4096, DEV)
    ns2["TOTAL_PINNED_MEMORY"] = 4096
    assert ns2["unpin_memory"](FT2(0x7F99_0000_1000, 4096)) is False
    assert ns2["TOTAL_PINNED_MEMORY"] == 4096 and ns2["PINNED_MEMORY"], (
        "production lost the record on a failing unregister")


def test_sensitivity_sync_failure_still_unregistering_is_detected():
    """If someone reorders to `unregister` outside the try, the fail-closed rule dies."""
    naive = _mm_variant(lambda s: s.replace(
        "            logging.warning(\"Unpin skipped, device synchronization failed: {}\".format(exc))\n"
        "            return False",
        "            logging.warning(\"Unpin skipped, device synchronization failed: {}\".format(exc))"))
    kinds, ok, ns, _ = _run_unpin_on(naive, sync_boom=True)
    assert "cudaHostUnregister" in kinds, "variant did not take effect as intended"
    # the production source must NOT have that property
    prod_kinds, prod_ok, prod_ns, _ = _run_unpin_on(MM.read_text(), sync_boom=True)
    assert "cudaHostUnregister" not in prod_kinds, (
        f"production unregisters after a failed sync: {prod_kinds}")
    assert prod_ns["PINNED_MEMORY"], "production dropped the record after a failed sync"


# ------------------------------------------------------------------ before/after, on disk


def _git_show(rev, relpath):
    return subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{rev}:{relpath}"],
                          capture_output=True, text=True, check=True).stdout


def test_head_and_worktree_are_graded_consistently():
    """The red-to-green indicator, written so it stays valid after the patch is committed.

    Phase 56 first expressed this as a parametrised pair over `git show HEAD:` and the
    working tree, which silently assumed HEAD was still unpatched. Once the candidate is
    committed -- the point of a merge -- both sides carry the sync and the "HEAD must not
    synchronise" case becomes a false expectation about history, not about the code.
    What is actually durable, and what this now asserts, is: for ANY revision of
    `model_management.py` reachable here, the observed event order must agree with whether
    that revision's `unpin_memory` contains a pre-unregister synchronisation. A revision
    that synchronises and a revision that does not must both be graded correctly; when the
    two sides are the same text there is no pair to compare and the test says so.
    """
    import hashlib
    import subprocess

    def events_for(source_text, tag):
        tree = ast.parse(source_text)
        fn = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "unpin_memory"]
        assert len(fn) == 1, f"{tag}: cannot locate exactly one unpin_memory"
        events = []
        ns, FakeTensor = base._events_namespace(events)
        ptr = 0x7F88_0000_0000
        body = ast.get_source_segment(source_text, fn[0])
        # seed the ledger in the shape THIS revision reads, not the one this file assumes
        ns["PINNED_MEMORY"][ptr] = (4096, DEV) if "pin_device" in body else 4096
        exec(compile(body, f"<{tag}>", "exec"), ns)
        events.clear()
        ns["unpin_memory"](FakeTensor(ptr, 4096))
        return [e[0] for e in events]

    worktree = MM.read_text()
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "show", "HEAD:comfy/model_management.py"],
                          capture_output=True, text=True).stdout
    assert head, "cannot read HEAD's copy; this test needs a git checkout to mean anything"
    same = hashlib.sha256(head.encode()).hexdigest() == hashlib.sha256(worktree.encode()).hexdigest()
    if same:
        pytest.skip("HEAD and the working tree are the same revision; there is no "
                    "before/after pair to grade here -- run it on a tree where the "
                    "candidate is uncommitted to exercise the flip")

    for source_text, tag in ((head, "HEAD"), (worktree, "worktree")):
        body = ast.get_source_segment(source_text, [n for n in ast.parse(source_text).body
                              if isinstance(n, ast.FunctionDef) and n.name == "unpin_memory"][0])
        claims_to_sync = "torch.cuda.synchronize" in body
        observed = "torch.cuda.synchronize" in events_for(source_text, tag)
        assert claims_to_sync == observed, (
            f"{tag}: source has a pre-unregister sync={claims_to_sync} but the executed "
            f"event order says {observed} -- the detector and the code disagree")


# --------------------------------------------------------------------- SDMA0 boundary


def test_nothing_here_establishes_a_sdma0_link():
    """Guard against the reading this phase must not produce.

    The candidate fixes an ordering contract. The two observed faults put their VAs in
    anonymous `rw-p` regions (Phase 47: 60 MiB; Phase 53: ~1.24 GiB, dev=00:00 inode=0), and
    `PINNED_MEMORY` is an in-process dict no out-of-process collector can read, so no test in
    this repository can associate a fault VA with a registration. Asserting the gap is closed
    would be the error, not a pass."""
    blob = (MM.read_text() + MP.read_text())
    for needle in ("sdma", "gfxhub", "page fault", "GCVM_L2"):
        assert needle not in blob.lower(), (
            f"{needle!r} now appears in the pin/unpin code, which would silently claim a "
            f"causal link no evidence supports")
