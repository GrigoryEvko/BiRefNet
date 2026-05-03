"""Regression test for the Swin BasicLayer SW-MSA attention-mask cache.

The previous implementation rebuilt the attention mask on every forward —
zeros allocation + 9-region slice writes + window_partition + masked_fill.
We now cache up to 4 entries keyed by (Hp, Wp, dtype, device).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


def _make_layer(dim=8, depth=2, num_heads=2, window_size=7):
    from models.backbones.swin_v1 import BasicLayer
    return BasicLayer(dim=dim, depth=depth, num_heads=num_heads,
                      window_size=window_size).eval()


def test_attn_mask_cache_reused_across_calls():
    layer = _make_layer()
    H, W = 28, 28  # multiple of window_size=7 → Hp=Wp=28
    x = torch.zeros(1, H * W, 8)
    with torch.no_grad():
        layer(x, H, W)
    # First call populated the cache exactly once.
    assert len(layer._attn_mask_cache) == 1
    cached = next(iter(layer._attn_mask_cache.values()))
    with torch.no_grad():
        layer(x, H, W)
    # Same key → same tensor object (no rebuild).
    assert next(iter(layer._attn_mask_cache.values())) is cached
    assert len(layer._attn_mask_cache) == 1


def test_attn_mask_cache_distinct_keys_for_distinct_shapes():
    layer = _make_layer()
    x_a = torch.zeros(1, 28 * 28, 8)
    x_b = torch.zeros(1, 35 * 35, 8)
    with torch.no_grad():
        layer(x_a, 28, 28)
        layer(x_b, 35, 35)
    assert len(layer._attn_mask_cache) == 2


def test_attn_mask_cache_evicts_oldest_at_5th_unique_shape():
    layer = _make_layer()
    shapes = [(28, 28), (35, 35), (42, 42), (49, 49), (56, 56)]
    with torch.no_grad():
        for H, W in shapes:
            x = torch.zeros(1, H * W, 8)
            layer(x, H, W)
    # Cap is 4; oldest got evicted.
    assert len(layer._attn_mask_cache) == 4
    keys = list(layer._attn_mask_cache.keys())
    # The first inserted (28×28) should no longer be in the cache.
    assert not any(k[0] == 28 and k[1] == 28 for k in keys)


def test_attn_mask_cache_lru_keeps_active_size():
    """LRU semantics: a recently-used entry should NOT be evicted just
    because it was inserted earlier. Insert 4 shapes, re-touch the first,
    insert a 5th — the second-inserted should be evicted, not the first.
    """
    layer = _make_layer()
    cpu = torch.device("cpu")
    keys_used = []
    for H, W in [(28, 28), (35, 35), (42, 42), (49, 49)]:
        layer._get_attn_mask(H, W, torch.float32, cpu)
        keys_used.append((H, W))
    # Re-touch the first shape — under LRU this should bump it to most-recent.
    layer._get_attn_mask(28, 28, torch.float32, cpu)
    # Now insert a 5th shape, triggering eviction.
    layer._get_attn_mask(56, 56, torch.float32, cpu)
    keys = list(layer._attn_mask_cache.keys())
    # 28×28 should still be present (most-recently-used). 35×35 (second-
    # inserted) is now the LRU and should have been evicted.
    sizes = {(k[0], k[1]) for k in keys}
    assert (28, 28) in sizes, "LRU evicted a recently-used entry"
    assert (35, 35) not in sizes, "LRU should have evicted second-inserted (oldest by access)"


def test_attn_mask_cache_distinct_dtype_keys():
    """Direct cache exercise: same (H, W) but different dtype → different key."""
    layer = _make_layer()
    H, W = 28, 28
    cpu = torch.device("cpu")
    layer._get_attn_mask(H, W, torch.float32, cpu)
    layer._get_attn_mask(H, W, torch.bfloat16, cpu)
    assert len(layer._attn_mask_cache) == 2


def test_attn_mask_avoids_neg_inf_under_bf16():
    """bf16 has no inf representation — using float('-inf').to(bf16) saturates
    to the dtype min, and a softmax over an all-min row would NaN. We now use
    finfo.min/2 which stays representable.
    """
    layer = _make_layer()
    H, W = 28, 28
    cpu = torch.device("cpu")
    mask = layer._get_attn_mask(H, W, torch.bfloat16, cpu)
    assert torch.isfinite(mask).all(), "attn_mask contains non-finite values under bf16"
    # The most-negative entry is finfo.min/2 — still representable.
    bf16_min = torch.finfo(torch.bfloat16).min
    assert mask.min().item() >= bf16_min, "mask min underflowed below bf16 representable range"


def test_attn_mask_cache_not_in_state_dict():
    """The plain dict must not be serialized into state_dict."""
    layer = _make_layer()
    H, W = 28, 28
    with torch.no_grad():
        layer(torch.zeros(1, H * W, 8), H, W)
    sd = layer.state_dict()
    # The cache key isn't there. (state_dict only contains parameters/buffers.)
    for k in sd:
        assert "attn_mask_cache" not in k
