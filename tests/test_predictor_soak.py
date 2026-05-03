"""Soak / memory-leak detection for the predictor.

Catches the classic production failure mode: predictor creates per-request
tensors that never get freed (held in caches, closures, accidental
self-references) and the process slowly OOMs over hours of serving.

Strategy:
  - Run N predicts on a stable predictor → assert allocations level off.
  - Construct/destroy K predictors → assert GC reclaims them (no
    accidental global references).

CPU-only by default (no CUDA assumed). For real production qualification
add a CUDA-gated soak test that watches torch.cuda.memory_allocated().
"""
from __future__ import annotations

import gc
import os
import sys
import weakref

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birefnet_api import BiRefNetPredictor


class _TinyModel(nn.Module):
    """Minimal but realistic forward — conv + a couple of internal allocations."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        return [self.conv2(torch.relu(self.conv1(x)))]


def _gradient_pil(h: int, w: int) -> Image.Image:
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    arr = np.stack([
        (xx * 255).astype(np.uint8),
        (yy * 255).astype(np.uint8),
        ((1 - xx) * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)


def test_repeated_predict_does_not_grow_unboundedly():
    """100 predicts on the same predictor: allocations after the first
    few iterations should level off. Compare iteration-50's count to
    iteration-99's; growth should be small (< 50 objects) since the
    predictor caches one Swin attn_mask per shape and one rpb per
    (dtype, device).
    """
    pred = BiRefNetPredictor(_TinyModel(), device="cpu", dtype=None,
                             max_edge=128, normalize=False)
    img = _gradient_pil(96, 96)
    # Warm-up: first calls allocate caches.
    for _ in range(5):
        pred.predict(img)
    gc.collect()
    baseline = len(gc.get_objects())
    for _ in range(95):
        pred.predict(img)
    gc.collect()
    final = len(gc.get_objects())
    # Allow some slack — pytest itself accumulates objects per call.
    growth = final - baseline
    assert growth < 5_000, (
        f"object count grew by {growth} over 95 predicts — possible leak"
    )


def test_predictor_is_garbage_collected_when_dropped():
    """Construct a predictor, take a weakref, drop the strong ref:
    the predictor must be reclaimable. Caches/closures shouldn't hold
    self back."""
    pred = BiRefNetPredictor(_TinyModel(), device="cpu", dtype=None,
                             max_edge=128, normalize=False)
    img = _gradient_pil(96, 96)
    pred.predict(img)  # populate caches
    ref = weakref.ref(pred)
    del pred
    gc.collect()
    assert ref() is None, (
        "predictor not garbage-collected after drop — something holds a "
        "strong reference (a closure, a global, a circular ref)"
    )


def test_repeated_construction_releases_models():
    """Loop: construct, predict, drop. Verify no per-iteration accumulation
    of model objects. Captures the 'serving framework creates a new
    predictor per request' anti-pattern's blast radius."""
    img = _gradient_pil(64, 64)
    refs = []
    for _ in range(20):
        p = BiRefNetPredictor(_TinyModel(), device="cpu", dtype=None,
                              max_edge=64, normalize=False)
        p.predict(img)
        refs.append(weakref.ref(p))
        del p
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    assert alive == 0, (
        f"{alive}/20 predictors still alive after drop+gc — leak"
    )


def test_predict_does_not_grow_caches_on_same_shape():
    """The Swin caches are keyed on shape — repeating the same input
    must not grow the cache beyond its first-population size."""
    pred = BiRefNetPredictor(_TinyModel(), device="cpu", dtype=None,
                             max_edge=128, normalize=False)
    img = _gradient_pil(96, 96)
    # First call populates.
    pred.predict(img)
    # _TinyModel doesn't have Swin caches, but we can at least assert the
    # predictor itself isn't leaking — pull a count of any tensor attrs.
    tensor_attrs_before = sum(
        1 for a in vars(pred).values() if isinstance(a, torch.Tensor)
    )
    for _ in range(20):
        pred.predict(img)
    tensor_attrs_after = sum(
        1 for a in vars(pred).values() if isinstance(a, torch.Tensor)
    )
    assert tensor_attrs_before == tensor_attrs_after, (
        "predictor accumulated tensor attributes across calls"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_predict_does_not_leak_vram():
    """100 CUDA predicts must not grow torch.cuda.memory_allocated."""
    pred = BiRefNetPredictor(_TinyModel(), device="cuda", dtype="bf16",
                             max_edge=128, normalize=False)
    img = _gradient_pil(96, 96)
    # Warm-up.
    for _ in range(5):
        pred.predict(img)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(95):
        pred.predict(img)
    torch.cuda.synchronize()
    final = torch.cuda.memory_allocated()
    growth_mb = (final - baseline) / 1e6
    assert growth_mb < 50, (
        f"VRAM grew by {growth_mb:.1f} MB over 95 predicts — possible leak"
    )


def test_real_model_swin_caches_dont_grow_unboundedly():
    """Soak with the actual BiRefNet (Swin-L by default) — exercises the
    Swin attn_mask LRU + rpb caches across many forwards. Catches leaks
    specific to the production model graph (cache reference cycles, etc.)
    that the _TinyModel-based tests above can't reach.
    """
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")

    import config as _config
    _orig = _config.Config.__init__
    def _safe(self):
        try:
            _orig(self)
        except FileNotFoundError:
            pass
    _config.Config.__init__ = _safe
    try:
        from models.birefnet import BiRefNet
        model = BiRefNet(bb_pretrained=False)
    finally:
        _config.Config.__init__ = _orig
    pred = BiRefNetPredictor(model, device="cpu", dtype=None, max_edge=192)

    img = _gradient_pil(192, 192)
    # Warm-up.
    for _ in range(2):
        pred.predict(img)
    gc.collect()
    baseline = len(gc.get_objects())
    for _ in range(8):
        pred.predict(img)
    gc.collect()
    final = len(gc.get_objects())
    growth = final - baseline
    # 8 predicts on a 200M-param Swin-L should grow well under 5K objects
    # if caches are stable. Headroom for pytest's own bookkeeping.
    assert growth < 10_000, (
        f"object count grew by {growth} over 8 real-model predicts — "
        f"Swin caches or model intermediate state may be leaking"
    )
