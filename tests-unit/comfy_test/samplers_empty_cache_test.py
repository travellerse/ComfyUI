"""Contract tests for `COMFY_EMPTY_CACHE_EVERY` (Q21-OPT-060).

What this file is responsible for, and what it deliberately is not:

  * It pins the **default**: with the variable unset (or empty, or `"0"`) the sampling loop
    must not call `torch.cuda.empty_cache()` at all. That is the shipped configuration, and
    the whole reason the hook is opt-in -- releasing the empty segments every step throws
    away the pool the next step would reuse.
  * It pins the cadence: `=N` releases on the Nth, 2Nth, ... call of `sampling_function`,
    counted across calls, not per sampler instance.
  * It pins the device guard: an uninitialised CUDA runtime must not be touched, because
    `empty_cache()` on a process that never created a context is a wasted driver round trip
    at best.

It does not measure the thing the hook exists for. The 2/2 pin failures -> 50/50 completion
result at 1920x1440 is evidence in report 62 H.3; reproducing it needs the display's scanout
allocation, which no test can arrange.

CPU only: `torch.cuda.is_initialized` and `torch.cuda.empty_cache` are both replaced, and the
cond-batch step is stubbed out, so nothing reaches a device.
"""

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.samplers as samplers  # noqa: E402


class _EmptyCacheRecorder:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


def _drive(monkeypatch, steps, empty_every=None, cuda_initialised=True):
    """Call `sampling_function` `steps` times with the cond-batch and CFG stages stubbed out."""
    recorder = _EmptyCacheRecorder()
    monkeypatch.setattr(torch.cuda, "empty_cache", recorder)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialised)
    monkeypatch.setattr(samplers, "_sampling_step_count", 0)
    monkeypatch.setattr(samplers, "cfg_function", lambda *a, **k: torch.zeros(1))

    if empty_every is None:
        monkeypatch.delenv("COMFY_EMPTY_CACHE_EVERY", raising=False)
    else:
        monkeypatch.setenv("COMFY_EMPTY_CACHE_EVERY", empty_every)

    model_options = {
        "sampler_calc_cond_batch_function": lambda a: [torch.zeros(1), torch.zeros(1)],
    }
    for _ in range(steps):
        samplers.sampling_function(
            None, torch.zeros(1), torch.zeros(1), None, torch.zeros(1), 1.0,
            model_options=model_options,
        )
    return recorder


def test_unset_never_releases(monkeypatch):
    assert _drive(monkeypatch, steps=3).calls == 0


def test_zero_and_empty_mean_off(monkeypatch):
    assert _drive(monkeypatch, steps=3, empty_every="0").calls == 0
    assert _drive(monkeypatch, steps=3, empty_every="").calls == 0


def test_releases_on_the_configured_cadence(monkeypatch):
    """Four steps at N=2 must release exactly twice -- on step 2 and step 4."""
    assert _drive(monkeypatch, steps=4, empty_every="2").calls == 2


def test_one_releases_every_step(monkeypatch):
    assert _drive(monkeypatch, steps=3, empty_every="1").calls == 3


def test_uninitialised_cuda_is_not_touched(monkeypatch):
    assert _drive(monkeypatch, steps=4, empty_every="1", cuda_initialised=False).calls == 0


def test_the_counter_is_module_level_not_per_sampler(monkeypatch):
    """The cadence has to survive across calls, otherwise N=2 would release on every call."""
    recorder = _EmptyCacheRecorder()
    monkeypatch.setattr(torch.cuda, "empty_cache", recorder)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(samplers, "_sampling_step_count", 0)
    monkeypatch.setattr(samplers, "cfg_function", lambda *a, **k: torch.zeros(1))
    monkeypatch.setenv("COMFY_EMPTY_CACHE_EVERY", "3")

    model_options = {
        "sampler_calc_cond_batch_function": lambda a: [torch.zeros(1), torch.zeros(1)],
    }
    for _ in range(2):
        samplers.sampling_function(
            None, torch.zeros(1), torch.zeros(1), None, torch.zeros(1), 1.0,
            model_options=model_options,
        )
    assert recorder.calls == 0
    samplers.sampling_function(
        None, torch.zeros(1), torch.zeros(1), None, torch.zeros(1), 1.0,
        model_options=model_options,
    )
    assert recorder.calls == 1
    assert samplers._sampling_step_count == 3
