"""Dispatch and correctness tests for the Qwen-Image-2.1 Edit text-encoder attention policy.

Two things are pinned here. First the scope: the policy is only allowed to replace the two
backends that materialise a whole score matrix, only above the size gate, and only on the
two models the Qwen-Image-2.1 text encoder attaches it to. Second the arithmetic: the
language model's GQA and its additive mask are load-bearing, and the vision tower runs at
fp32 with head_dim 72, which is not the geometry any accelerated backend handles. The
substituted backend is checked against an fp64 reference, not against itself.
"""

import functools
import inspect
import os

import pytest
import torch

import comfy.ops
from comfy.ldm.modules import attention as attn
from comfy.text_encoders import qwen_image21
from comfy.text_encoders.llama import Llama2_, Llama2Config
from comfy.text_encoders.qwen35 import Qwen35VisionModel

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VISION_CFG = dict(hidden_size=8, num_heads=2, intermediate_size=16, depth=1, patch_size=16,
                  temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
                  num_position_embeddings=4, out_hidden_size=8)


def tiny_vision_model():
    model = Qwen35VisionModel(VISION_CFG, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
    # `manual_cast` leaves the parameters uninitialised on purpose - they are meant to be
    # overwritten by a checkpoint - so give them values before running the tower.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    return model


def tiny_language_model():
    config = Llama2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, vocab_size=16, rms_norm_eps=1e-6)
    config.head_dim = 4
    return Llama2_(config, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)


# --------------------------------------------------------------------------- policy semantics
def test_basic_backend_is_replaced_by_a_pinned_sub_quad():
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    pinned = policy.apply(attn.attention_basic, 32, 2434, 2434, 4)
    assert isinstance(pinned, functools.partial)
    assert pinned.func is attn.attention_sub_quad
    assert pinned.keywords == {"query_chunk_size": 512, "inplace_softmax": True}


def test_sub_quad_backend_is_pinned_too():
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    pinned = policy.apply(attn.attention_sub_quad, 16, 6144, 6144, 4)
    assert pinned.func is attn.attention_sub_quad
    assert pinned.keywords["query_chunk_size"] == 512


@pytest.mark.parametrize("backend", [
    attn.attention_pytorch,
    attn.attention_split,
    attn.attention_sage,
    attn.attention_flash,
])
def test_an_explicitly_chosen_backend_is_left_alone(backend):
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    assert policy.apply(backend, 32, 2434, 2434, 4) is backend


def test_a_score_matrix_below_the_gate_is_left_alone():
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512, min_score_bytes=128 * 2 ** 20)
    # 32 x 1024 x 1024 x 4 B = 128 MiB exactly, so one token less must not qualify.
    assert policy.apply(attn.attention_basic, 32, 1024, 1024, 4) is not attn.attention_basic
    assert policy.apply(attn.attention_basic, 32, 1024, 1023, 4) is attn.attention_basic


def test_the_gate_is_measured_in_bytes_not_tokens():
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512, min_score_bytes=128 * 2 ** 20)
    # Same shapes, half the element size: below the gate.
    assert policy.apply(attn.attention_basic, 32, 1024, 1024, 2) is attn.attention_basic


def test_default_gate_has_no_size_condition():
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    assert policy.min_score_bytes == 0
    assert policy.apply(attn.attention_basic, 1, 1, 1, 4) is not attn.attention_basic


def test_policy_does_not_forward_the_fused_hook():
    # The fused kernel declines this geometry anyway, but a policy that quietly opted into
    # it would make the transient depend on a second mechanism.
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    pinned = policy.apply(attn.attention_basic, 32, 2434, 2434, 4)
    assert "fused_attention" not in pinned.keywords


# ---------------------------------------------------------------------------------- defaults
def test_vision_tower_defaults_to_no_policy():
    assert tiny_vision_model().attention_policy is None


def test_language_model_defaults_to_no_policy():
    assert tiny_language_model().attention_policy is None


def test_the_two_consumers_read_the_policy_they_are_given():
    for module in (Qwen35VisionModel, Llama2_):
        source = inspect.getsource(module.forward)
        assert "attention_policy" in source, module.__name__


# --------------------------------------------------------------------------- vision consumer
def test_vision_tower_hands_the_policy_its_geometry():
    seen = []

    class Recording:
        def apply(self, attention_function, heads, q_tokens, k_tokens, element_size):
            seen.append((heads, q_tokens, k_tokens, element_size))
            return attention_function

    model = tiny_vision_model()
    model.attention_policy = Recording()
    x = torch.randn(4, 3 * 2 * 16 * 16)
    model(x, torch.tensor([[1, 2, 2]]))
    assert seen == [(2, 4, 4, 4)]


def test_vision_output_is_unchanged_by_the_policy():
    torch.manual_seed(0)
    x = torch.randn(4, 3 * 2 * 16 * 16)
    grid = torch.tensor([[1, 2, 2]])
    plain = tiny_vision_model()
    with torch.inference_mode():
        reference = plain(x, grid)
    assert torch.isfinite(reference).all()
    plain.attention_policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    with torch.inference_mode():
        pinned = plain(x, grid)
    assert torch.allclose(reference, pinned, atol=1e-6)


# ------------------------------------------------------------------ Qwen-Image-2.1 scoping
def test_policy_constants_match_the_measured_geometry():
    assert qwen_image21.VISION_ATTENTION.query_chunk_size == 512
    assert qwen_image21.VISION_ATTENTION.min_score_bytes == 0
    assert qwen_image21.LM_ATTENTION.query_chunk_size == 512
    assert qwen_image21.LM_ATTENTION.min_score_bytes == 128 * 2 ** 20
    # The edit conditioning runs the language model at 2434 tokens with 32 heads, i.e.
    # 723 MiB, and the plain text-to-image path at 29 tokens, i.e. 105 KiB.
    assert qwen_image21.LM_ATTENTION.apply(attn.attention_basic, 32, 2434, 2434, 4) is not attn.attention_basic
    assert qwen_image21.LM_ATTENTION.apply(attn.attention_basic, 32, 29, 29, 4) is attn.attention_basic


def test_only_the_qwen_image21_text_encoder_constructs_a_policy():
    # qwen35.py and llama.py only name the type in a comment; constructing one is what
    # scopes it, and only the Qwen-Image-2.1 text encoder is allowed to.
    directory = os.path.dirname(qwen_image21.__file__)
    owners = sorted(name for name in os.listdir(directory)
                    if name.endswith(".py") and "FixedChunkAttentionPolicy(" in
                    open(os.path.join(directory, name)).read())
    assert owners == ["qwen_image21.py"]


# -------------------------------------------------------------------------------- LM numerics
LM = dict(heads=32, kv_heads=8, head_dim=128, dtype=torch.float32)


def _lm_inputs(seq, heads=LM["heads"], kv_heads=LM["kv_heads"], head_dim=LM["head_dim"],
               dtype=LM["dtype"], device="cuda"):
    torch.manual_seed(7)
    q = torch.randn(1, heads, seq, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, kv_heads, seq, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, kv_heads, seq, head_dim, dtype=dtype, device=device)
    # Additive mask as Llama2_.forward builds it: 0 where allowed, finfo.min/4 where not,
    # plus a causal part, and a couple of fully masked padding rows.
    neg = torch.finfo(dtype).min / 4
    keep = torch.ones(1, seq, dtype=dtype, device=device)
    keep[0, -2:] = 0.0
    mask = 1.0 - keep.reshape(1, 1, -1, seq).expand(1, 1, seq, seq)
    mask = mask.masked_fill(mask.to(torch.bool), neg)
    mask += torch.empty(seq, seq, dtype=dtype, device=device).fill_(neg).triu_(1)
    return q, k, v, mask


def _fp64_reference(q, k, v, mask, chunk=256):
    heads, kv_heads = q.shape[1], k.shape[1]
    k = k.repeat_interleave(heads // kv_heads, dim=1).double()
    v = v.repeat_interleave(heads // kv_heads, dim=1).double()
    scale = q.shape[-1] ** -0.5
    out = []
    for start in range(0, q.shape[2], chunk):
        end = min(start + chunk, q.shape[2])
        sim = torch.einsum("b h i d, b h j d -> b h i j", q[:, :, start:end].double(), k) * scale
        sim = sim + mask[:, :, start:end].double()
        out.append(torch.einsum("b h i j, b h j d -> b h i d", sim.softmax(dim=-1), v))
    return torch.cat(out, dim=2).permute(0, 2, 1, 3).reshape(1, q.shape[2], -1)


@needs_gpu
def test_lm_swap_matches_basic_and_the_fp64_reference():
    q, k, v, mask = _lm_inputs(1024)
    with torch.inference_mode():
        reference = _fp64_reference(q, k, v, mask)
        basic = attn.attention_basic(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)
        policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
        pinned = policy.apply(attn.attention_basic, LM["heads"], 1024, 1024, q.element_size())
        swapped = pinned(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)

    assert swapped.shape == basic.shape == (1, 1024, LM["heads"] * LM["head_dim"])
    assert swapped.dtype == q.dtype
    assert torch.isfinite(swapped).all()
    ref_max = reference.abs().max()
    for name, out in (("attention_basic", basic), ("pinned sub_quad", swapped)):
        rel = ((out - reference).abs().max() / ref_max).item()
        assert rel < 1e-5, (name, rel)
    # And the two backends agree with each other to fp32 summation order.
    assert ((swapped - basic).abs().max() / basic.abs().max()).item() < 1e-5


@needs_gpu
def test_lm_gqa_is_preserved():
    q, k, v, mask = _lm_inputs(256)
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=128)
    pinned = policy.apply(attn.attention_basic, LM["heads"], 256, 256, q.element_size())
    with torch.inference_mode():
        swapped = pinned(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)
        # The same call with the kv heads expanded by the caller: if the swap dropped GQA
        # or repeated the wrong dimension this would not line up.
        explicit = attn.attention_basic(
            q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1), LM["heads"],
            mask=mask, skip_reshape=True)
    assert torch.allclose(swapped, explicit, atol=1e-5, rtol=1e-4)


@needs_gpu
def test_lm_additive_mask_is_applied():
    q, k, v, mask = _lm_inputs(256)
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=128)
    pinned = policy.apply(attn.attention_basic, LM["heads"], 256, 256, q.element_size())
    with torch.inference_mode():
        swapped = pinned(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)
        # Same inputs with the blocked keys physically removed from the last query row's
        # view: a mask that is added rather than applied would differ here.
        keep = mask[0, 0, -1] == 0
        row = 255
        allowed = torch.nonzero(keep).flatten()
        k_rep = k.repeat_interleave(4, dim=1)
        v_rep = v.repeat_interleave(4, dim=1)
        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h d, b h j d -> b h j", q[:, :, row], k_rep[:, :, allowed]) * scale
        reference = torch.einsum("b h j, b h j d -> b h d", sim.softmax(-1), v_rep[:, :, allowed])
    assert torch.allclose(swapped[0, row].reshape(LM["heads"], LM["head_dim"]), reference[0],
                          atol=1e-5, rtol=1e-4)


@needs_gpu
def test_lm_transient_at_the_edit_geometry_is_under_the_basic_one():
    q, k, v, mask = _lm_inputs(2434)
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    pinned = policy.apply(attn.attention_basic, LM["heads"], 2434, 2434, q.element_size())
    with torch.inference_mode():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        attn.attention_basic(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)
        torch.cuda.synchronize()
        basic_mib = (torch.cuda.max_memory_allocated() - base) / 2 ** 20

        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        pinned(q, k, v, LM["heads"], mask=mask, skip_reshape=True, enable_gqa=True)
        torch.cuda.synchronize()
        pinned_mib = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    # 723 MiB of score plus 723 MiB of softmax against one 159 MiB query chunk.
    assert basic_mib > 1400, basic_mib
    assert pinned_mib < 400, pinned_mib


# ---------------------------------------------------------------------------- vision numerics
@needs_gpu
def test_vision_geometry_takes_the_pinned_chunk_and_keeps_its_values():
    heads, head_dim, tokens = 16, 72, 3072
    torch.manual_seed(11)
    q = torch.randn(1, heads, tokens, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(1, heads, tokens, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(1, heads, tokens, head_dim, dtype=torch.float32, device="cuda")
    policy = attn.FixedChunkAttentionPolicy(query_chunk_size=512)
    pinned = policy.apply(attn.attention_sub_quad, heads, tokens, tokens, q.element_size())
    assert pinned.func is attn.attention_sub_quad
    with torch.inference_mode():
        reference = attn.attention_sub_quad(q, k, v, heads, skip_reshape=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        pinned_out = pinned(q, k, v, heads, skip_reshape=True)
        torch.cuda.synchronize()
        pinned_mib = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    assert pinned_out.shape == reference.shape
    assert torch.allclose(pinned_out, reference, atol=1e-5, rtol=1e-4)
    # 16 x 512 x 3072 x 4 B = 96 MiB for the score, written over in place.
    assert pinned_mib < 160, pinned_mib
