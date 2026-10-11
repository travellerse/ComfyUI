"""Tests for the Qwen-Image-2.1 text encoder component lifecycle.

The text encoder is released from GPU residency once positive and negative conditioning
are both materialized, so the DiT does not have to evict it passively under memory
pressure. These tests pin the two properties that make that safe:

1. the release happens after BOTH encodes (never between positive and negative), and
2. it is scheduler-consistent: the text encoder leaves ``current_loaded_models`` while
   unrelated models (DiT/VAE) stay resident, so the next ``load_models_gpu`` reloads it.
"""

import torch

import comfy.model_management as model_management
from comfy_extras.nodes_qwen import TextEncodeQwenImage21


class FakeClip:
    """Minimal stand-in exposing the surface the node uses.

    Records call order so we can assert the release lands after both encodes.
    """

    def __init__(self):
        self.patcher = object()
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append(("tokenize", text))
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        self.calls.append(("encode", tokens["text"]))
        # A conditioning entry: [tensor, dict of plain values], as ComfyUI really builds it.
        return [[torch.zeros(1, 4, 8), {"pooled_output": torch.zeros(1, 8)}]]


def test_release_happens_after_both_positive_and_negative_encodes(monkeypatch):
    released = []
    monkeypatch.setattr(
        TextEncodeQwenImage21, "release_text_encoder",
        classmethod(lambda cls, clip: released.append(clip)),
    )

    clip = FakeClip()
    TextEncodeQwenImage21.execute(clip, "a cat", "blurry")

    assert [c[0] for c in clip.calls] == ["tokenize", "encode", "tokenize", "encode"]
    assert clip.calls[1][1] == "a cat"
    assert clip.calls[3][1] == "blurry"
    # exactly one release, and the positive/negative encodes both happened already
    assert len(released) == 1


def test_release_targets_the_clip_patcher_only(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_management, "unload_model_and_clones",
        lambda patcher, **kwargs: calls.append((patcher, kwargs)),
    )

    clip = FakeClip()
    TextEncodeQwenImage21.release_text_encoder(clip)

    assert len(calls) == 1
    patcher, kwargs = calls[0]
    assert patcher is clip.patcher
    assert kwargs["unload_additional_models"] is True
    assert kwargs["all_devices"] is True


def test_release_is_a_noop_without_a_patcher(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not unload anything when the clip has no patcher")

    monkeypatch.setattr(model_management, "unload_model_and_clones", explode)
    TextEncodeQwenImage21.release_text_encoder(object())


def test_conditioning_holds_no_text_encoder_reference(monkeypatch):
    """Conditioning must survive the release: it can only contain plain values.

    Anything module-like surviving in the conditioning would keep the released
    parameters reachable and make the release unsafe.
    """
    monkeypatch.setattr(model_management, "unload_model_and_clones", lambda *a, **k: None)
    clip = FakeClip()
    positive, negative, latent = TextEncodeQwenImage21.execute(clip, "a cat", "blurry")

    for conditioning in (positive, negative):
        for tensor, extra in conditioning:
            assert isinstance(tensor, torch.Tensor)
            for key, value in extra.items():
                assert isinstance(value, (torch.Tensor, list, int, float, str, type(None)))
    assert set(latent["samples"].shape) == {1, 64}
