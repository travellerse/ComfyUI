"""Branch semantics of the Qwen-Image-2.1 text-encoder release, under fakes only.

What this can and cannot say: it pins how ``TextEncodeQwenImage21.release_text_encoder`` and
``model_management.unload_model_and_clones`` decide what to keep, for constructed inputs.
It does NOT show what the production run's ``clip.patcher``, ``clone_base_uuid`` or live
tensor references were - that stays a runtime question.

Two implementation facts read from the current tree, which is why the assertions look the way
they do:

  * ``unload_model_and_clones`` compares ``model.clone_base_uuid`` values to build its
    keep-list, but ``free_memory`` then filters with ``shift_model not in keep_loaded`` -
    i.e. membership is by LoadedModel *object identity*, so these tests assert on objects.
    (comfy/model_management.py:893-908, unload_model_and_clones above it)
  * the real ``free_memory`` calls ``get_free_memory(device)`` and walks the manager, so it is
    replaced wholesale here; nothing in this file touches a device, loads a model or allocates
    a GPU tensor.

The sibling test ``qwen_te_release_scheduler_test.py`` already covers the same path with real
ModelPatcher objects; this file adds the structural branches (uuid mismatch, empty manager,
None entries, the two flags) instead of duplicating it.
"""

import types

import pytest
import torch

# Same guard the sibling release test uses: comfy.model_management resolves the torch device
# at import (comfy/model_management.py:364 -> get_torch_device), which raises on a host with
# no visible GPU. Declaring cpu mode first keeps this module importable with the device
# hidden, so the file is not accidentally GPU-dependent.
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as mm  # noqa: E402
from comfy_extras.nodes_qwen import TextEncodeQwenImage21  # noqa: E402

CPU = torch.device("cpu")
CPU2 = torch.device("cpu")


class Entry:
    """One current_loaded_models element, with just the fields the code reads."""

    def __init__(self, model):
        self.model = model


class Model:
    def __init__(self, uuid):
        self.clone_base_uuid = uuid


class Patcher(Model):
    def __init__(self, uuid, additional=()):
        super().__init__(uuid)
        self._additional = list(additional)
        self.additional_calls = 0

    def get_nested_additional_models(self):
        self.additional_calls += 1
        return list(self._additional)


@pytest.fixture
def spy(monkeypatch):
    """Replace the manager's global state for one test and restore it afterwards."""
    calls = []
    entries = []

    def fake_free_memory(memory_required, device, keep_loaded=(), *a, **kw):
        calls.append({"need": memory_required, "device": device,
                      "keep": list(keep_loaded)})

    monkeypatch.setattr(mm, "free_memory", fake_free_memory)
    monkeypatch.setattr(mm, "current_loaded_models", entries)
    monkeypatch.setattr(mm, "get_all_torch_devices", lambda: [CPU, CPU2])
    monkeypatch.setattr(mm, "get_torch_device", lambda: CPU)
    yield types.SimpleNamespace(calls=calls, entries=entries)
    assert mm.free_memory is fake_free_memory, "fixture must still own the patch on teardown"


# --- release_text_encoder -----------------------------------------------------------------

def test_no_patcher_never_reaches_the_unload_path(spy):
    TextEncodeQwenImage21.release_text_encoder(types.SimpleNamespace(patcher=None))
    assert spy.calls == []


def test_patcher_without_attribute_is_treated_as_absent(spy):
    TextEncodeQwenImage21.release_text_encoder(types.SimpleNamespace())
    assert spy.calls == []


def test_matching_uuid_is_excluded_from_keep_loaded_others_are_kept(spy):
    te = Entry(Model("TE"))
    dit = Entry(Model("DIT"))
    vae = Entry(Model("VAE"))
    spy.entries.extend([te, dit, vae])

    TextEncodeQwenImage21.release_text_encoder(
        types.SimpleNamespace(patcher=Patcher("TE")))

    assert len(spy.calls) == 2, "all_devices=True must run one free pass per torch device"
    for call in spy.calls:
        assert call["need"] == 1e30
        # identity, not uuid: free_memory filters with `not in keep_loaded`
        assert te not in call["keep"], "the released text encoder must be dropped from keep"
        assert dit in call["keep"] and vae in call["keep"]


def test_empty_manager_still_calls_free_memory_with_an_empty_keep_list(spy):
    TextEncodeQwenImage21.release_text_encoder(
        types.SimpleNamespace(patcher=Patcher("TE")))
    assert [c["keep"] for c in spy.calls] == [[], []]


def test_uuid_mismatch_keeps_the_target_resident(spy):
    """The inert shape: a patcher whose uuid matches nothing leaves everything kept.

    A release call that reaches free_memory with a full keep-list does not unload the text
    encoder, which is the failure mode worth being able to recognise from logs alone.
    """
    te = Entry(Model("LOADER-UUID"))
    spy.entries.append(te)

    TextEncodeQwenImage21.release_text_encoder(types.SimpleNamespace(patcher=Patcher("TE-UUID")))

    assert len(spy.calls) == 2
    for call in spy.calls:
        assert call["keep"] == [te], "no uuid match means nothing gets excluded"


def test_entry_without_a_model_object_is_kept(spy):
    """`if loaded_model.model is not None` guards the uuid comparison, so None entries stay."""
    none_entry = Entry(None)
    te = Entry(Model("TE"))
    spy.entries.extend([none_entry, te])

    TextEncodeQwenImage21.release_text_encoder(types.SimpleNamespace(patcher=Patcher("TE")))

    for call in spy.calls:
        assert none_entry in call["keep"]
        assert te not in call["keep"]


def test_nested_additional_models_are_excluded_too(spy):
    clone = Model("CLONE")
    te = Entry(Model("TE"))
    te_clone = Entry(clone)
    other = Entry(Model("OTHER"))
    spy.entries.extend([te, te_clone, other])

    TextEncodeQwenImage21.release_text_encoder(
        types.SimpleNamespace(patcher=Patcher("TE", additional=[clone])))

    for call in spy.calls:
        assert te not in call["keep"] and te_clone not in call["keep"]
        assert call["keep"] == [other]


# --- unload_model_and_clones directly -----------------------------------------------------

def test_all_devices_false_frees_once_on_the_torch_device(spy):
    te = Entry(Model("TE"))
    spy.entries.append(te)
    mm.unload_model_and_clones(Patcher("TE"), unload_additional_models=True, all_devices=False)
    assert len(spy.calls) == 1
    assert spy.calls[0]["device"] == CPU


def test_all_devices_true_frees_on_every_device(spy):
    spy.entries.append(Entry(Model("TE")))
    mm.unload_model_and_clones(Patcher("TE"), unload_additional_models=True, all_devices=True)
    assert [c["device"] for c in spy.calls] == [CPU, CPU2]


def test_flag_off_does_not_enumerate_additional_models(spy):
    clone_model = Model("CLONE")
    p = Patcher("TE", additional=[clone_model])
    te = Entry(Model("TE"))
    clone_entry = Entry(clone_model)
    spy.entries.extend([te, clone_entry])

    mm.unload_model_and_clones(p, unload_additional_models=False, all_devices=True)

    assert p.additional_calls == 0, "the nested lookup must be skipped when the flag is off"
    for call in spy.calls:
        assert te not in call["keep"]
        assert clone_entry in call["keep"], "without the flag a nested model stays resident"
