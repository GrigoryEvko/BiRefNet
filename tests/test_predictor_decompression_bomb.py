"""Tests for the decompression-bomb guard.

PIL's MAX_IMAGE_PIXELS is set to None at dataset-import time, disabling
the built-in protection. The predictor's per-instance max_pixels cap is
the actual security boundary for user-uploaded inputs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birefnet_api import BiRefNetPredictor
from birefnet_api.predictor import _load_pil, _check_pixel_budget


class _Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1 / 3)

    def forward(self, x):
        return [self.conv(x)]


def test_check_pixel_budget_passes_within_limit():
    _check_pixel_budget(100, 100, max_pixels=20_000)  # 10_000 < 20_000


def test_check_pixel_budget_passes_when_none():
    """max_pixels=None disables the check entirely."""
    _check_pixel_budget(100_000, 100_000, max_pixels=None)


def test_check_pixel_budget_raises_with_dimensions_in_message():
    with pytest.raises(ValueError, match="100x200.*max_pixels=10,000"):
        _check_pixel_budget(200, 100, max_pixels=10_000)


def test_load_pil_rejects_oversized_pil_input():
    """Passing an already-decoded oversized PIL.Image still gets rejected."""
    img = Image.new("RGB", (1000, 1000), color=(0, 0, 0))
    with pytest.raises(ValueError, match="exceed max_pixels"):
        _load_pil(img, max_pixels=100_000)  # 1M > 100K


def test_load_pil_rejects_oversized_numpy_before_decode():
    """Numpy input larger than max_pixels: rejected before any conversion work."""
    arr = np.zeros((2000, 2000, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _load_pil(arr, max_pixels=1_000_000)


def test_load_pil_rejects_oversized_path_via_header(tmp_path):
    """Path input: PIL.Image.open is lazy — header read tells us the size
    without paying the decode cost. Rejection happens before .load()."""
    p = tmp_path / "big.png"
    Image.new("RGB", (2000, 2000), color=(0, 0, 0)).save(p)
    with pytest.raises(ValueError, match="exceed max_pixels"):
        _load_pil(str(p), max_pixels=1_000_000)
    # The file should still be readable afterwards (header-read doesn't
    # corrupt anything, the FD is properly closed by the context manager).
    assert os.path.exists(p)


def test_load_pil_rejects_oversized_tensor():
    """Tensor input also goes through the budget check after CHW->HWC."""
    t = torch.zeros(3, 2000, 2000, dtype=torch.uint8)
    with pytest.raises(ValueError):
        _load_pil(t, max_pixels=1_000_000)


def test_predictor_default_cap_blocks_50k_input():
    """Default 200MP cap rejects a 50K×50K malicious upload."""
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=64)
    arr = np.zeros((50_000, 50_000, 3), dtype=np.uint8)  # 2.5GP, well above 200MP
    with pytest.raises(ValueError, match="exceed max_pixels"):
        pred.predict(arr)


def test_predictor_default_cap_accepts_12k_input():
    """Default 200MP cap leaves room for the 12K user-cutout workflow.
    We use a sentinel-sized array (just to test the budget check), not an
    actual 12K render — the test runs on every commit and HR allocations
    would be too expensive."""
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=64)
    # 12000×8000 = 96MP, comfortably within 200MP default. Verify the check
    # alone passes; we don't actually run the model on this huge array (slow).
    _check_pixel_budget(8000, 12000, pred.max_pixels)


def test_predictor_max_pixels_can_be_disabled():
    """max_pixels=None: no cap (back-compat for trusted inputs)."""
    pred = BiRefNetPredictor(
        _Identity(), device="cpu", dtype=None, normalize=False,
        max_edge=64, max_pixels=None,
    )
    assert pred.max_pixels is None
    arr = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # Should not raise even though it would under the default cap.
    pred.predict(arr)


def test_predictor_normal_input_passes_default_cap():
    """A 256×256 image is way under the cap and should produce a mask."""
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=64)
    img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    mask = pred.predict(img)
    assert mask.shape == (256, 256)
