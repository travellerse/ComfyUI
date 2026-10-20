"""Contract tests for the gfx103x key repack (Q21-OPT-066 / OPT-067).

What this file is responsible for, and what it deliberately is not:

  * It pins the **gate**. With `Q21_TRITON_REPACK_KEY` unset the helper must return `None`
    before it loads Triton or allocates anything, because that is the production default and
    an import there would be paid by every attention call in every workflow.
  * It pins the two decline paths the caller depends on: no usable kernel (Triton missing or
    the import raised) and a head dimension the 32-wide tile cannot cover. Both must return
    `None` and bump `skipped` -- `attention_sub_quad` reads that `None` as "keep the strided
    operand", so raising here would turn a slow path into a crash.
  * It pins the failure path: a kernel that raises is swallowed into the same `None`, with a
    warning, and counted as skipped.

It does not run the kernel. The 1.242x / 1.417x call-level numbers are measured evidence in
the Triton tree (`docs/rdna2/performance/rdna2-key-repack-improvement.md`) and the model-level
image gate is still BLOCKED, so a benchmark here would only re-derive a number it cannot
validate. It also does not assert the packed buffer's contents -- that needs a device; the
kernel is pinned as bit-exact against `torch.contiguous()` in that report.

CPU only: nothing below allocates on or launches to a device.
"""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.modules.attention as attn  # noqa: E402


class _RecordingKernel:
    """Stands in for the JIT function: records whether the launch site was reached."""

    def __init__(self):
        self.calls = 0

    def __getitem__(self, grid):
        def launch(*a, **k):
            self.calls += 1

        return launch


class _RaisingKernel:
    def __getitem__(self, grid):
        def launch(*a, **k):
            raise RuntimeError("synthetic launch failure")

        return launch


class _StubTriton:
    @staticmethod
    def cdiv(a, b):
        return -(-a // b)


@pytest.fixture(autouse=True)
def _fresh_repack_stats(monkeypatch):
    monkeypatch.setattr(attn, "_repack_stats", {"applied": 0, "skipped": 0})
    monkeypatch.setattr(attn, "_repack_state", None)


def _key(shape=(2, 72, 128)):
    return torch.zeros(shape, dtype=torch.float16)


def test_default_gate_declines_before_loading_triton(monkeypatch):
    """The shipped default must cost nothing -- not even an import attempt."""

    def _must_not_run():
        raise AssertionError("the flag is off, so Triton must not be loaded")

    monkeypatch.setattr(attn, "TRITON_REPACK_KEY", False)
    monkeypatch.setattr(attn, "_load_repack_kernel", _must_not_run)

    assert attn.repack_key_packed(_key(), dim_head=128, k_tokens=128) is None
    assert attn._repack_stats == {"applied": 0, "skipped": 0}


def test_missing_kernel_is_counted_as_skipped(monkeypatch):
    monkeypatch.setattr(attn, "TRITON_REPACK_KEY", True)
    monkeypatch.setattr(attn, "_load_repack_kernel", lambda: (None, None, "triton unavailable"))

    assert attn.repack_key_packed(_key(), dim_head=128, k_tokens=128) is None
    assert attn._repack_stats["skipped"] == 1
    assert attn._repack_stats["applied"] == 0


def test_head_dim_not_divisible_by_the_tile_declines(monkeypatch):
    """72 is the Qwen3-VL vision tower's head_dim: 72 % 32 == 8, so the tile cannot cover it."""
    kernel = _RecordingKernel()
    monkeypatch.setattr(attn, "TRITON_REPACK_KEY", True)
    monkeypatch.setattr(attn, "_load_repack_kernel", lambda: (_StubTriton, kernel, None))

    assert attn.repack_key_packed(_key(), dim_head=72, k_tokens=128) is None
    assert kernel.calls == 0
    assert attn._repack_stats["skipped"] == 1


def test_launch_failure_falls_back_to_the_strided_operand(monkeypatch, caplog):
    monkeypatch.setattr(attn, "TRITON_REPACK_KEY", True)
    monkeypatch.setattr(attn, "_load_repack_kernel", lambda: (_StubTriton, _RaisingKernel(), None))

    with caplog.at_level("WARNING"):
        assert attn.repack_key_packed(_key(), dim_head=128, k_tokens=128) is None

    assert attn._repack_stats["skipped"] == 1
    assert attn._repack_stats["applied"] == 0
    assert any("key repack failed" in r.message for r in caplog.records)


def test_a_covered_head_dim_reaches_the_launch_site(monkeypatch):
    """Negative control for the two decline tests: 128 % 32 == 0 must actually try."""
    kernel = _RecordingKernel()
    monkeypatch.setattr(attn, "TRITON_REPACK_KEY", True)
    monkeypatch.setattr(attn, "_load_repack_kernel", lambda: (_StubTriton, kernel, None))

    out = attn.repack_key_packed(_key(), dim_head=128, k_tokens=128)

    assert kernel.calls == 1
    assert attn._repack_stats["applied"] == 1
    assert tuple(out.shape) == (2, 128, 128)
