"""Release must leave ComfyUI's scheduler state consistent with the real ModelPatcher.

This is a CPU-only test using real ``ModelPatcher`` objects: after the text encoder is
released it must be gone from ``current_loaded_models`` (so the next ``load_models_gpu``
reloads it instead of believing it is still resident), while unrelated models must stay
resident. It also covers a second acquire/execute/release cycle in the same process.
"""

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as model_management  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402
from comfy_extras.nodes_qwen import TextEncodeQwenImage21  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8, bias=False)


@pytest.fixture(autouse=True)
def release_on_this_device(monkeypatch):
    """Pin the release device to CPU so the test exercises the real code path on any host.

    ``unload_model_and_clones(all_devices=True)`` walks ``get_all_torch_devices()``, which
    is the GPU list on a GPU host. Pinning it keeps this test CPU-only and deterministic.
    """
    cpu = torch.device("cpu")
    monkeypatch.setattr(model_management, "get_all_torch_devices", lambda *a, **k: [cpu])


def _patcher():
    return ModelPatcher(Tiny(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def _resident(patcher):
    return any(lm.model is patcher for lm in model_management.current_loaded_models)


def test_release_removes_text_encoder_but_keeps_other_models_resident():
    text_encoder = _patcher()
    dit = _patcher()
    vae = _patcher()
    try:
        # DiT/VAE acquire first, then the text encoder encodes (all three resident).
        model_management.load_models_gpu([dit, vae, text_encoder])
        assert _resident(text_encoder) and _resident(dit) and _resident(vae)

        TextEncodeQwenImage21.release_text_encoder(SimpleNamespace(patcher=text_encoder))

        assert not _resident(text_encoder), "text encoder must leave GPU residency"
        assert _resident(dit), "DiT must stay resident"
        assert _resident(vae), "VAE must stay resident"
    finally:
        model_management.current_loaded_models.clear()


def test_text_encoder_can_be_acquired_again_after_release():
    """A second execution in the same process must reload cleanly, not double-release."""
    text_encoder = _patcher()
    try:
        model_management.load_models_gpu([text_encoder])
        TextEncodeQwenImage21.release_text_encoder(SimpleNamespace(patcher=text_encoder))
        assert not _resident(text_encoder)

        # run 2: the scheduler must see it as absent and load it again
        model_management.load_models_gpu([text_encoder])
        assert _resident(text_encoder)

        TextEncodeQwenImage21.release_text_encoder(SimpleNamespace(patcher=text_encoder))
        assert not _resident(text_encoder)
    finally:
        model_management.current_loaded_models.clear()


def test_releasing_when_not_loaded_is_a_noop():
    never_loaded = _patcher()
    other = _patcher()
    try:
        model_management.load_models_gpu([other])
        TextEncodeQwenImage21.release_text_encoder(SimpleNamespace(patcher=never_loaded))
        assert _resident(other)
    finally:
        model_management.current_loaded_models.clear()
