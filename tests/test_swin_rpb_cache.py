"""Regression tests for the relative_position_bias eval-time cache.

The Swin window-attention bias was computed every forward as
`bias_table[index].view(N,N,-1).permute(...).unsqueeze(0).to(dtype, device)`
— a small allocation + dtype copy on every block, every forward.

In eval mode the table is fixed, so we cache by (dtype, device). Training
always rebuilds (table updates each step). The cache also clears on a
.train() transition.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


def _make_attn(window=7, num_heads=2, dim=8):
    from models.backbones.swin_v1 import WindowAttention
    return WindowAttention(dim=dim, window_size=(window, window), num_heads=num_heads)


def test_rpb_cache_populated_in_eval():
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)  # (B_, H, N, Dh) — N=49 for window=7
    attn._relative_position_bias(q)
    assert len(attn._rpb_cache) == 1


def test_rpb_cache_skipped_in_train():
    attn = _make_attn().train()
    q = torch.zeros(1, 2, 49, 4)
    attn._relative_position_bias(q)
    # No cache populated in train.
    assert len(attn._rpb_cache) == 0


def test_rpb_cache_cleared_on_train_transition():
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    attn._relative_position_bias(q)
    assert len(attn._rpb_cache) == 1
    attn.train()
    # Cache emptied; the table is about to start changing.
    assert len(attn._rpb_cache) == 0


def test_rpb_cache_returns_same_tensor_on_repeat():
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    a = attn._relative_position_bias(q)
    b = attn._relative_position_bias(q)
    assert a is b


def test_rpb_cache_distinct_dtype_keys():
    attn = _make_attn().eval()
    q32 = torch.zeros(1, 2, 49, 4, dtype=torch.float32)
    q_bf16 = torch.zeros(1, 2, 49, 4, dtype=torch.bfloat16)
    attn._relative_position_bias(q32)
    attn._relative_position_bias(q_bf16)
    assert len(attn._rpb_cache) == 2


def test_rpb_cache_not_in_state_dict():
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    attn._relative_position_bias(q)
    keys = attn.state_dict().keys()
    assert all("rpb_cache" not in k for k in keys)


def test_rpb_cache_cleared_on_load_state_dict():
    """The relative_position_bias_table changes when a checkpoint is loaded.
    Cached gathered+cast tensors derived from the OLD table are now stale —
    cache must clear so the next forward rebuilds against the new weights.
    """
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    cached_first = attn._relative_position_bias(q).clone()

    # Build a second module with a different bias table, copy its state.
    attn2 = _make_attn().eval()
    with torch.no_grad():
        attn2.relative_position_bias_table.fill_(0.5)  # distinctive value
    sd = attn2.state_dict()

    attn.load_state_dict(sd)
    # Cache must be empty after load_state_dict.
    assert len(attn._rpb_cache) == 0
    # Next forward returns bias derived from the NEW table, not the cached old one.
    new_bias = attn._relative_position_bias(q)
    assert not torch.allclose(new_bias, cached_first), (
        "rpb_cache returned stale bias after load_state_dict"
    )
