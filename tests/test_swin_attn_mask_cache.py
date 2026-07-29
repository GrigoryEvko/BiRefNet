"""Regression test for the Swin BasicLayer SW-MSA attention-mask cache.

The previous implementation rebuilt the attention mask on every forward —
zeros allocation + 9-region slice writes + window_partition + masked_fill.
We now cache up to 4 entries keyed by (Hp, Wp, dtype, device).
"""
from __future__ import annotations

import math
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


def test_attn_mask_sentinel_is_bounded():
    """The masked-out sentinel must be SMALL, not near the dtype maximum.

    This test previously asserted the opposite, on a false premise: "bf16 has no
    inf representation — float('-inf').to(bf16) saturates to the dtype min".
    bfloat16 has fp32's 8 exponent bits and represents inf natively:

        torch.tensor(float('-inf')).to(torch.bfloat16)  ->  -inf

    The sentinel that premise justified, finfo.min/2 = -1.69e38, has no headroom
    for the log2(e) rescaling fused attention kernels apply when folding softmax
    into exp2 — (m + m) * 1.4427 overflows fp32 to -inf, and one -inf minus -inf
    in the row-max subtraction NaNs the entire row. That NaN reaches the output,
    survives clamp(0, 1), and casts to an all-zero uint8 mask, which is how
    eu-north-1 served fully transparent mattes for six days with HTTP 200 and a
    healthy /health.

    So: still finite (the original property, kept), but now also bounded well
    away from the dtype range so no kernel rescaling can overflow it.
    """
    layer = _make_layer()
    H, W = 28, 28
    cpu = torch.device("cpu")
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        mask = layer._get_attn_mask(H, W, dtype, cpu)
        assert torch.isfinite(mask).all(), f"attn_mask non-finite under {dtype}"
        m = mask.min().item()
        assert m < 0.0, f"{dtype}: nothing was masked"

        # HEADROOM. The sentinel must survive what a fused attention kernel does
        # to it: it gets added to the score (and possibly to a second mask term)
        # and the whole sum is rescaled by log2(e) to fold softmax into exp2.
        # Require room for 8x plus that rescale before reaching the dtype limit.
        # Stated against the dtype's own range, so this is meaningful for fp16
        # (max 65504) as well as for bf16/fp32 (max ~3.4e38) — an absolute
        # threshold, or one phrased as "N orders of magnitude below finfo.min",
        # would be vacuous for one and impossible for the other.
        headroom = abs(m) * 8.0 * 1.4426950408889634
        assert headroom < abs(torch.finfo(dtype).min), (
            f"{dtype}: sentinel {m:.3e} has no headroom — 8x plus the log2(e) "
            f"rescale reaches {headroom:.3e} against finfo.min "
            f"{torch.finfo(dtype).min:.3e}, which overflows to -inf and NaNs "
            f"the softmax row"
        )

        # STILL FULLY MASKS. exp(sentinel) must be negligible against the
        # unmasked weights, or the mask is not doing its job.
        assert math.exp(m) < 1e-20, (
            f"{dtype}: sentinel {m:.3e} leaves exp(m)={math.exp(m):.3e} of "
            f"weight on masked entries"
        )


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
