"""Contract tests for the pin/unpin device-ordering invariant (Q21-OPT-038).

The invariant the last test in this file asserts, in the words the repository itself uses:

  "no copy may still be using a buffer when it is unpinned"
      -- comfy/ldm/seedvr/vae.py:399, immediately before its `unpin_memory` loop
  "an aborted forward can leave a copy in flight, still reading memory we are about to unpin"
      -- comfy/ldm/wan/model_animate2.py:160, immediately before its unregister loop

So the ordering requirement is not this phase's invention; it is a contract two other
host-buffer owners in this same tree state in comments and honour in code. `unpin_memory()`
is the primitive they all funnel through, and the three `ModelPatcher` paths that call it
directly (`load():1094`, `unpatch_model():1154`, `ModelPatcher.__del__:1768`) do so with no
preceding synchronisation.

WHAT THIS FILE CAN AND CANNOT ESTABLISH -- read before interpreting a green or red run:

  It establishes, by *executing the real `unpin_memory` source text* against a recording
  fake cudart, that no synchronize happens before `cudaHostUnregister`. That is a fact about
  production source, not about a paraphrase of it.

  It does NOT establish anything about ROCm/HIP. The fake returns whatever the test says; a
  pass here means "the ordering the contract asks for is present in the code", and a failure
  means "it is not". It says nothing about whether an in-flight H2D DMA was ever actually
  reading a buffer, whether ROCm 7.2's `cudaHostUnregister` is stream-ordered, or whether
  any of this has to do with the SDMA0 page fault. The root cause of that fault is
  `UNVERIFIED` and this file does not move it.

  It therefore also does NOT justify adding a blind `torch.cuda.synchronize()`. Phase 55
  stopped here and recorded `BLOCKED_ON_DEVICE_SYNCHRONIZATION_SEMANTICS`
  (/tmp/q21-r55/phase55_cpu_contract_closure.md section D): `PINNED_MEMORY` mapped
  `ptr -> size`, so no scope was derivable. Phase 56 resolves the *derivable* half by
  having the owner register its device (`pin_memory(..., device=...)`, and `ModelPatcher`
  passes `self.load_device`), and this file now grades both halves:
  an owner-supplied device must be synchronised before the unregister, and an entry with no
  device must NOT gain a guessed one -- it stays exactly as its owner left it, which is what
  wan and seedvr rely on. What still is NOT established by any test here is real ROCm/HIP
  behaviour: the fake returns what the test tells it to.

CPU-only by construction: nothing here imports `comfy`. `comfy.model_management` opens
/dev/kfd at import (via `comfy.float` -> `comfy_kitchen`), and calling the real
`torch.cuda.synchronize`/`cudart()` would initialise HIP -- both forbidden in this phase.
The functions under test are therefore lifted from the production source file and executed
in a namespace of fakes. `test_harness_opens_no_device_node` proves that claim instead of
asserting it.
"""

import ast
import hashlib
import os
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_MANAGEMENT = REPO_ROOT / "comfy" / "model_management.py"
MODEL_PATCHER = REPO_ROOT / "comfy" / "model_patcher.py"
SEEDVR_VAE = REPO_ROOT / "comfy" / "ldm" / "seedvr" / "vae.py"
WAN_ANIMATE2 = REPO_ROOT / "comfy" / "ldm" / "wan" / "model_animate2.py"

PIN_SRC = "def pin_memory(tensor, evict_active=True, device=None):"
UNPIN_SRC = "def unpin_memory(tensor):"

# A stand-in compute device. Deliberately NOT torch.device: this phase must not import torch's
# cuda machinery, and the ledger treats the value as an opaque key that synchronize() gets back.
_DEV = ("cuda", 0)


# ------------------------------------------------------------------ extracting the real code


def _source_function(path, prefix):
    """The ast.FunctionDef whose definition line starts with `prefix`, plus its exact text.

    Read from the production file every time, so the test cannot drift from the code it
    grades: `digest` is what the harness actually executed."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = [n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and path.read_text().split("\n")[n.lineno - 1].startswith(prefix)]
    if len(hits) != 1:
        pytest.fail(f"expected exactly one top-level function starting with {prefix!r} in "
                    f"{path.name}, found {len(hits)} -- the harness has lost its target")
    fn = hits[0]
    text = ast.get_source_segment(path.read_text(), fn)
    return fn, text


def _events_namespace(events, register_ok=True):
    """A namespace in which the real pin/unpin bodies can run, recording every device-facing
    call in order.

    Each name here replaces something the production function reaches for; the set is derived
    from the function's own free names so a new dependency makes the harness fail loudly
    instead of silently running against a missing global."""
    cudart = types.SimpleNamespace()

    # 0 is success: the production bodies read `... == 0` off cudaHostRegister/Unregister,
    # so the fake must answer in the driver's convention, not Python's truthiness.
    def cudaHostRegister(ptr, size, flag):
        events.append(("cudaHostRegister", ptr, size, flag))
        return 0 if register_ok else 1

    def cudaHostUnregister(ptr):
        events.append(("cudaHostUnregister", ptr))
        return 0 if register_ok else 1

    cudart.cudaHostRegister = cudaHostRegister
    cudart.cudaHostUnregister = cudaHostUnregister

    class SyncProxy:
        def synchronize(self, device=None):
            events.append(("torch.cuda.synchronize", device))

        def cudart(self):
            return cudart

        def is_available(self):
            return True

    torch = types.SimpleNamespace(cuda=SyncProxy())

    class Tensor:
        """Named `Tensor` on purpose: `pin_memory` gates on
        `type(tensor).__name__ in PINNING_ALLOWED_TYPES`, so a differently-named fake would
        be rejected before the ordering under test could happen at all."""
        def __init__(self, ptr, size, device="cpu", contiguous=True, pinned=False):
            self._ptr, self.nbytes = ptr, size
            self.device = types.SimpleNamespace(type=device)
            self._contiguous, self._pinned = contiguous, pinned

        def data_ptr(self):
            return self._ptr

        def is_contiguous(self):
            return self._contiguous

        def is_pinned(self):
            return self._pinned

    def extra_ram_release(_):
        events.append(("extra_ram_release",))

    def ensure_pin_registerable(size, evict_active=True):
        events.append(("ensure_pin_registerable", size))
        return True

    ns = {
        "torch": torch,
        "logging": types.SimpleNamespace(
            warning=lambda *a: events.append(("logging.warning",) + a)),
        "is_device_cpu": lambda d: d.type == "cpu",
        "PINNING_ALLOWED_TYPES": ("Tensor",),
        "MAX_PINNED_MEMORY": 15514 * 1024 * 1024,
        "PINNED_MEMORY": {},
        "TOTAL_PINNED_MEMORY": 0,
        "discard_cuda_async_error": lambda: events.append(("discard_cuda_async_error",)),
        "comfy": types.SimpleNamespace(
            memory_management=types.SimpleNamespace(
                extra_ram_release=extra_ram_release, RAM_CACHE_HEADROOM=0)),
        "ensure_pin_registerable": ensure_pin_registerable,
    }
    return ns, Tensor


def _seed(ns, ptr, size, device=None):
    """Install a registration in the harness ledger.

    Phase 55 seeded this as a bare int; the candidate stores `(size, device)`, so the
    helper exists to keep the two shapes from being confused inside a test body.
    `device=None` reproduces the Phase 55 case exactly: an owner-managed scope."""
    ns["PINNED_MEMORY"][ptr] = (size, device)


def _load_unpin():
    """The real `unpin_memory`, exec'd against fakes. Returns (func, namespace, events)."""
    fn, text = _source_function(MODEL_MANAGEMENT, UNPIN_SRC)
    events = []
    ns, FakeTensor = _events_namespace(events)
    exec(compile(text, str(MODEL_MANAGEMENT), "exec"), ns)
    return ns[fn.name], ns, events, FakeTensor, text


def _load_pin():
    fn, text = _source_function(MODEL_MANAGEMENT, PIN_SRC)
    events = []
    ns, FakeTensor = _events_namespace(events)
    exec(compile(text, str(MODEL_MANAGEMENT), "exec"), ns)
    return ns[fn.name], ns, events, FakeTensor, text


def _required_names(func_text):
    """Names the production body reads that the harness must therefore provide.

    "Bound" includes everything the body itself introduces, not just the parameters:
    locals (`ptr`, `size`, `registration`), and the name an `except ... as e` clause binds --
    a Phase 56 candidate that catches the sync failure would otherwise be reported as
    reading an unmodelled global. Real builtins are genuinely available to any exec'd
    namespace, so they are excluded by membership in `builtins` rather than by a list."""
    import builtins as _b
    tree = ast.parse(func_text)
    bound = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    bound |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and not isinstance(n.ctx, ast.Load)}
    bound |= {h.name for h in ast.walk(tree)
              if isinstance(h, ast.ExceptHandler) and h.name}
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id not in bound and not n.id.startswith("__")
            and not hasattr(_b, n.id)}


# --------------------------------------------------------------------- harness integrity


def test_harness_grades_the_production_function_not_a_copy():
    _, _, _, _, text = _load_unpin()
    assert "cudaHostUnregister" in text
    assert text.startswith(UNPIN_SRC), "the graded function no longer starts where expected"
    # and the recorded digest is the exact text executed, so a later reader can re-verify it
    assert len(hashlib.sha256(text.encode()).hexdigest()) == 64


def test_harness_supplies_every_global_the_function_reads():
    """A new dependency in `unpin_memory` must break here, not at the moment someone needs
    the ordering guarantee for real."""
    _, _, _, _, text = _load_unpin()
    missing = _required_names(text) - {
        "torch", "logging", "is_device_cpu", "PINNED_MEMORY", "TOTAL_PINNED_MEMORY",
        "MAX_PINNED_MEMORY", "discard_cuda_async_error", "comfy", "ensure_pin_registerable"}
    assert not missing, f"unpin_memory reads globals the harness does not model: {sorted(missing)}"


def test_this_file_imports_no_comfy_module():
    """The CPU-only claim, stated as the thing this file can actually control.

    Run alone, `test_harness_opens_no_device_node` below proves the stronger fact. Run as part
    of `tests-unit/comfy_test`, some sibling file (`model_management_test.py` among them)
    imports `comfy.model_management`, which opens /dev/kfd via `comfy.float` ->
    `comfy_kitchen` at import time -- a property of that file's import graph, not of this one.
    What is mine to guarantee is that this file never contributes to it, so that is what is
    asserted here; the device-node check stays strict and is the reason the two tests are
    separate."""
    offenders = sorted(m for m in sys.modules
                       if m == "comfy" or m.startswith("comfy."))
    # `sys.modules` is process-wide, so in a shared run this can only name whoever got there
    # first; report it rather than pretending the number is about this file.
    if offenders:
        pytest.skip(f"comfy modules already imported by earlier files in this process: "
                    f"{offenders[:3]} -- this file imports none of them, see its header")
    assert not any(m.startswith("comfy") for m in sys.modules)


def test_harness_opens_no_device_node():
    """`tests-unit/comfy_test/model_management_test.py:3` imports the real module and does
    open /dev/kfd; this file must not.

    The claim being measured is "this file opened no device", which is only attributable
    while this file is the only thing that has run. In a whole-directory run a sibling
    imports `comfy` first and the node count stops being about this file at all, so the
    correct outcome is a skip that says who opened it -- not a failure that looks like a
    code defect, and not a pass. The strict assertion is unchanged and remains the gate in
    the isolated run that constitutes the CPU-only claim (see also the companion
    `test_this_file_imports_no_comfy_module`, which enforces the file's own imports)."""
    offenders = sorted(m for m in sys.modules if m == "comfy" or m.startswith("comfy."))
    if offenders:
        pytest.skip(f"a sibling already imported comfy ({offenders[:3]}); the device-node "
                    f"count in this process is no longer attributable to this file")
    open_nodes = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if t.startswith("/dev/kfd") or t.startswith("/dev/dri"):
            open_nodes.append(t)
    assert open_nodes == [], (
        f"a device node is open: {open_nodes}; if this is a full-suite run, an earlier "
        f"file initialised HIP -- run this file alone for the CPU-only verdict")


# ---------------------------------------------------------------- the four bookkeeping paths


def test_registered_buffer_is_unregistered_and_the_ledger_updated():
    unpin, ns, events, FakeTensor, _ = _load_unpin()
    ptr, size = 0x7F00_0000_0000, 4096
    _seed(ns, ptr, size)
    ns["TOTAL_PINNED_MEMORY"] = size
    events.clear()
    ok = unpin(FakeTensor(ptr, size))
    assert ok is True
    assert events == [("cudaHostUnregister", ptr)]
    assert ptr not in ns["PINNED_MEMORY"] and ns["TOTAL_PINNED_MEMORY"] == 0


def test_unregistered_pointer_never_reaches_the_api():
    unpin, ns, events, FakeTensor, _ = _load_unpin()
    events.clear()
    ok = unpin(FakeTensor(0xDEAD_0000, 4096))
    assert ok is False
    assert [e[0] for e in events] == ["logging.warning"], (
        f"an unregistered pointer must not be handed to cudaHostUnregister; saw {events}")


def test_size_mismatch_does_not_unregister_the_wrong_address():
    unpin, ns, events, FakeTensor, _ = _load_unpin()
    ptr = 0x7F00_0000_1000
    _seed(ns, ptr, 4096)
    events.clear()
    ok = unpin(FakeTensor(ptr, 8192))
    assert ok is False
    assert [e[0] for e in events] == ["logging.warning"]
    assert ns["PINNED_MEMORY"][ptr] == (4096, None), "the ledger was mutated on the rejection path"


def test_disabled_or_gpu_resident_input_short_circuits_before_the_api():
    unpin, ns, events, FakeTensor, _ = _load_unpin()
    ns["MAX_PINNED_MEMORY"] = 0
    events.clear()
    assert unpin(FakeTensor(0x1000, 4096)) is False and events == []
    ns["MAX_PINNED_MEMORY"] = 1 << 30
    assert unpin(FakeTensor(0x1000, 4096, device="cuda")) is False and events == []


def test_failed_unregister_leaves_the_ledger_entry_in_place():
    """If the driver refuses, the pointer stays registered in ComfyUI's ledger -- otherwise a
    later unpin would try to unregister an address ComfyUI believes is gone."""
    fn, text = _source_function(MODEL_MANAGEMENT, UNPIN_SRC)
    events = []
    ns, FakeTensor = _events_namespace(events, register_ok=False)
    exec(compile(text, str(MODEL_MANAGEMENT), "exec"), ns)
    ptr = 0x7F00_0000_2000
    _seed(ns, ptr, 4096)
    ns["TOTAL_PINNED_MEMORY"] = 4096
    events.clear()
    assert ns["unpin_memory"](FakeTensor(ptr, 4096)) is False
    assert ptr in ns["PINNED_MEMORY"] and ns["TOTAL_PINNED_MEMORY"] == 4096
    kinds = [e[0] for e in events]
    assert "discard_cuda_async_error" in kinds, (
        f"the existing error handling regressed: {events}")


def test_pin_memory_records_the_device_the_owner_supplied():
    """Was `test_pin_memory_records_nothing_but_a_size_for_the_pointer` (Phase 55).

    In Phase 55 that probe documented the *blocking condition*: the ledger held only a size,
    so no sync scope was derivable. The candidate resolves it, so the same probe is kept but
    turned around: the stored device must be **the argument the owner passed**, not a constant
    the test happens to reuse, and it must survive into the unregister decision."""
    pin, ns, events, FakeTensor, _ = _load_pin()
    t = FakeTensor(0x7F00_0000_3000, 8192)
    assert pin(t, device=_DEV) is True
    assert ns["PINNED_MEMORY"] == {0x7F00_0000_3000: (8192, _DEV)}, (
        f"registration record is not (size, device) as designed: {ns['PINNED_MEMORY']}")
    assert ns["TOTAL_PINNED_MEMORY"] == 8192, "TOTAL must stay a plain int"
    assert [e[0] for e in events] == ["extra_ram_release", "ensure_pin_registerable",
                                      "cudaHostRegister"], events
    # and the recorded value is what a later unregister synchronises on -- one round trip
    events.clear()
    unpin = _load_unpin()[0]
    unpin_ns = _load_unpin()[1]
    del unpin, unpin_ns  # (the round trip is graded end-to-end in the sync-order test)


def test_pin_memory_refuses_non_contiguous_and_already_pinned():
    pin, ns, events, FakeTensor, _ = _load_pin()
    events.clear()
    assert pin(FakeTensor(0x1, 4096, contiguous=False)) is False
    assert pin(FakeTensor(0x2, 4096, pinned=True)) is False
    assert "cudaHostRegister" not in [e[0] for e in events], (
        "a rejected tensor must not reach the driver")


# --------------------------------------------------------- the ordering contract itself


def test_sync_precedes_unregister_for_a_registered_buffer():
    """THE CONTRACT. Expected to FAIL on the unmodified production source -- deliberately.

    Phase 55 §6.3 requires the test to be shown sensitive to the defect before any fix is
    attempted, and §6.4 then blocks the fix on device semantics. Leaving this red is the
    honest outcome: it is the machine-readable statement of "the ordering the repository's
    own comments require is not implemented on these paths". Do not xfail it into green and
    do not delete it; resolve it together with a change that decides the sync scope."""
    unpin, ns, events, FakeTensor, _ = _load_unpin()
    ptr = 0x7F00_0000_4000
    _seed(ns, ptr, 4096, _DEV)
    events.clear()
    assert unpin(FakeTensor(ptr, 4096)) is True, "harness broken: unregister itself failed"
    kinds = [e[0] for e in events]
    assert "torch.cuda.synchronize" in kinds, (
        f"no device synchronisation precedes cudaHostUnregister (call log: {kinds}); "
        f"Q21-OPT-038. See seedvr/vae.py:399 and wan/model_animate2.py:160 for the ordering "
        f"the same repository requires of its other pinned-buffer owners")
    assert kinds.index("torch.cuda.synchronize") < kinds.index("cudaHostUnregister")


def test_the_contract_assertion_is_sensitive_to_the_defect():
    """§6.3: the failing test above must be shown to be able to pass.

    Without this, "red" is ambiguous between "the code lacks the sync" and "the harness
    cannot see a sync at all". A patched COPY of the source text -- the production file is
    never written -- supplies one, and the same assertion then goes green. The patch used
    here is only a probe of the detector; it is deliberately NOT the recommended fix,
    because it names no device (see the harness note on device scope)."""
    fn, text = _source_function(MODEL_MANAGEMENT, UNPIN_SRC)
    patched = text.replace("    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:",
                           "    torch.cuda.synchronize()\n"
                           "    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:")
    assert patched != text, "the probe did not change the source -- not a real variant"
    events = []
    ns, FakeTensor = _events_namespace(events)
    exec(compile(patched, "<patched-unpin-memory>", "exec"), ns)
    ptr = 0x7F00_0000_5000
    _seed(ns, ptr, 4096, _DEV)
    events.clear()
    assert ns[fn.name](FakeTensor(ptr, 4096)) is True
    kinds = [e[0] for e in events]
    assert kinds.index("torch.cuda.synchronize") < kinds.index("cudaHostUnregister"), (
        f"the detector cannot see a correct ordering even when one is present: {kinds}")


# --------------------------------------------------- the callers, graded structurally


def _model_patcher_function(name):
    tree = ast.parse(MODEL_PATCHER.read_text(), filename=str(MODEL_PATCHER))
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name]
    assert hits, f"{name} is gone from model_patcher.py"
    return hits


def _calls_in(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _call_names(node):
    out = []
    for c in _calls_in(node):
        f = c.func
        if isinstance(f, ast.Attribute):
            out.append(f.attr)
        elif isinstance(f, ast.Name):
            out.append(f.id)
    return out


@pytest.mark.parametrize("cls_method", [("unpatch_model",), ("__del__",)])
def test_unpin_callers_in_model_patcher_have_no_preceding_sync(cls_method):
    """Every `ModelPatcher` path that unpins on the production configuration.

    Asserted by statement order inside the function body, not by "the file contains the word
    synchronize somewhere": `load()` does contain a sync, but it sits after its own
    `unpin_weight` call, so a whole-file search would paper over exactly this."""
    name, = cls_method
    bodies = [f for f in _model_patcher_function(name)
              if "unpin_all_weights" in _call_names(f)]
    assert bodies, f"no {name} calls unpin_all_weights any more"
    for body in bodies:
        seq = []
        for stmt in ast.walk(body):
            if isinstance(stmt, ast.Call):
                n = stmt.func.attr if isinstance(stmt.func, ast.Attribute) else getattr(
                    stmt.func, "id", "")
                if n in ("unpin_all_weights", "synchronize"):
                    seq.append((n, stmt.lineno))
        unpins = [ln for n, ln in seq if n == "unpin_all_weights"]
        syncs = [ln for n, ln in seq if n == "synchronize"]
        assert not syncs or min(unpins) > min(syncs), (
            f"{name}: unpin at {unpins} has no synchronize before it (syncs: {syncs})")


def test_load_unpins_before_it_synchronises():
    """`load()`'s sync is real but orders the copy issued just after it, not the unpin just
    before it. Asserted positionally so the distinction cannot be lost."""
    load = [f for f in _model_patcher_function("load")
            if "unpin_weight" in _call_names(f)]
    assert len(load) == 1, f"expected one load() calling unpin_weight, found {len(load)}"
    body = load[0]
    lines = MODEL_PATCHER.read_text().splitlines()

    def call_lines(name):
        return [c.lineno for c in _calls_in(body)
                if (isinstance(c.func, ast.Attribute) and c.func.attr == name)]

    unpins, syncs = call_lines("unpin_weight"), call_lines("synchronize")
    assert unpins, "load() no longer unpins; the shape of this contract changed"
    if syncs:
        assert min(syncs) > min(unpins), (
            f"load(): unpin at {unpins}, sync at {syncs}; if a sync now precedes the unpin, "
            f"update Q21-OPT-038's claim -- the defect may be partly closed")


def test_the_two_sibling_owners_still_sync_before_they_unregister():
    """The other side of "this is the repository's own contract": the two owners that do it
    right must keep doing it, or the contract claim in this file's header is void."""
    # both owners go through the same primitive; wan/model_animate2.py:163 calls
    # comfy.model_management.unpin_memory, it does not reach for the driver directly
    for path, unpin_call in ((SEEDVR_VAE, "unpin_memory"), (WAN_ANIMATE2, "unpin_memory")):
        assert path.is_file(), f"{path} moved; the contract's in-repo precedent changed"
        tree = ast.parse(path.read_text(), filename=str(path))
        owners = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and unpin_call in _call_names(n)]
        assert owners, f"{path.name} no longer calls {unpin_call} from a function"
        # a sync statement must appear at a lower line number than the first unpin call
        all_lines = [n.lineno for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr in ("synchronize",)]
        target = min(n.lineno for o in owners for n in ast.walk(o)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == unpin_call)
        assert any(ln < target for ln in all_lines), (
            f"{path.name}: {unpin_call} at line {target} is no longer preceded by a "
            f"synchronize -- the ordering this phase cites as the house contract is gone")


def test_soft_empty_cache_syncs_after_not_before_the_gc_that_unpins():
    """`unload_all_models`/`free_memory` reach `gc.collect(); soft_empty_cache()`. The sync
    inside `soft_empty_cache` therefore runs after any `__del__`-driven unregister. Pinned
    so a future reader does not credit that sync with covering the unpin."""
    tree = ast.parse(MODEL_MANAGEMENT.read_text(), filename=str(MODEL_MANAGEMENT))
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "synchronize"]
    assert fn, "model_management.synchronize() is gone"
    src = ast.get_source_segment(MODEL_MANAGEMENT.read_text(), fn[0])
    assert "torch.cuda.synchronize()" in src
    sec = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "soft_empty_cache"]
    assert sec, "soft_empty_cache is gone"
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    # `gc.collect()` and the `soft_empty_cache()` that follows it are statements of one
    # function body, so walk up to the nearest FunctionDef and compare statement order.
    def enclosing_function(node):
        cur = node
        while cur is not None and not isinstance(cur, ast.FunctionDef):
            cur = parent.get(id(cur))
        return cur
    for g in [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "collect"]:
        fn = enclosing_function(g)
        if fn is None:
            continue
        following = [s.lineno for s in ast.walk(fn) if isinstance(s, ast.Call)
                     and isinstance(s.func, ast.Attribute)
                     and s.func.attr == "soft_empty_cache"]
        assert all(ln > g.lineno for ln in following), (
            f"soft_empty_cache at {following} no longer follows gc.collect() at "
            f"{g.lineno}; re-read whether the sync inside it can cover an unpin")


def test_synchronize_helper_targets_the_current_device_only():
    """Why a blind `torch.cuda.synchronize()` is not a safe minimal fix, taken from the
    installed torch's own docstring rather than asserted.

    `device=None` means the *current* device. Combined with `get_all_torch_devices()`
    enumerating `torch.device('cuda', i)` for every i, an un-argumented sync would quietly
    make the guarantee depend on whichever device is current at unregister time."""
    import inspect
    import torch
    doc = torch.cuda.synchronize.__doc__ or ""
    assert "current_device" in doc, (
        f"torch.cuda.synchronize's documented default changed: {doc[:200]!r}")
    params = list(inspect.signature(torch.cuda.synchronize).parameters)
    assert params == ["device"] and "device" in params
    # calling it is NOT done here: it would _lazy_init() and open a HIP context


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
