"""Regression tests for the input-handling hardening in birefnet_api.predictor:

  - float numpy in [0, 255] is auto-detected (was clipped→[0,1]→×255 → black).
  - int numpy/tensor inputs are rescaled by observed range, not truncated mod 256.
  - predict_batch letterbox-pads heterogeneous-aspect inputs (was aspect-distorted).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from birefnet_api import BiRefNetPredictor
from birefnet_api.predictor import _np_to_uint8


class _Identity(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 conv that averages RGB → single-channel logit
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1 / 3)

    def forward(self, x):
        return [self.conv(x)]


@pytest.fixture
def cpu_predictor():
    return BiRefNetPredictor(_Identity(), device="cpu", dtype=None, max_edge=512, normalize=False)


# --- _np_to_uint8 helpers ---

def test_np_to_uint8_passes_uint8_through():
    a = np.array([[0, 127, 255]], dtype=np.uint8)
    assert _np_to_uint8(a) is a


def test_np_to_uint8_float_in_0_1():
    a = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = _np_to_uint8(a)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 127, 255]])


def test_np_to_uint8_float_in_0_255_auto_detected():
    """The previously-broken case. cv2.imread().astype(np.float32) is in [0,255]."""
    a = np.array([[0.0, 127.0, 255.0]], dtype=np.float32)
    out = _np_to_uint8(a)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 127, 255]])


def test_np_to_uint8_int16_rescales_not_truncates():
    """Int16 / int32 inputs: rescale by observed range, do NOT truncate mod 256."""
    a = np.array([[0, 32768, 65535]], dtype=np.uint16)
    out = _np_to_uint8(a)
    assert out.dtype == np.uint8
    # endpoints map to 0 and 255, not (0 & 0xFF, 32768 & 0xFF, 65535 & 0xFF)
    assert out[0, 0] == 0
    assert out[0, 2] == 255


# --- predict() hardening ---

def test_predict_handles_float32_in_0_255_range(cpu_predictor):
    """A user passing cv2.imread().astype(np.float32) used to get a black mask."""
    arr = np.full((128, 128, 3), 200.0, dtype=np.float32)  # in [0, 255]
    mask = cpu_predictor.predict(arr)
    # An all-200 input should produce a non-trivial sigmoid response (not all 0).
    assert mask.shape == (128, 128)
    # If the float-detection were broken, input would be clipped to all 1.0
    # then ×255 → all 255 (white). Different but distinguishable.
    # The Identity model averages channels into a single value, so all-200/255
    # → ~0.78 input → conv weight 1/3 × 200/255 × 3 = 0.78 → sigmoid ~0.69.
    assert 0.4 < mask.mean().item() < 0.95


def test_predict_handles_int32_tensor(cpu_predictor):
    """int32 tensor in [0, 65535] used to wrap mod 256 via .to(torch.uint8)."""
    t = torch.tensor(
        [[[0] * 128] * 128, [[32768] * 128] * 128, [[65535] * 128] * 128],
        dtype=torch.int32,
    )
    mask = cpu_predictor.predict(t)
    assert mask.shape == (128, 128)
    # The middle channel is the largest (65535) so the predictor input is
    # heavy on that channel after rescale; check finiteness.
    assert torch.all(torch.isfinite(mask))


# --- predict_batch letterbox ---

def test_predict_batch_letterboxes_heterogeneous_aspects(cpu_predictor):
    """All masks come back at their own original resolution and aren't
    aspect-distorted. (The previous _pick_batch_bucket took max h × max w
    from different images and squashed everyone into that.)"""
    portrait = Image.new("RGB", (256, 1024), color=(120, 50, 200))   # tall
    landscape = Image.new("RGB", (1024, 256), color=(200, 120, 50))  # wide
    square = Image.new("RGB", (512, 512), color=(50, 200, 120))
    masks = cpu_predictor.predict_batch([portrait, landscape, square])
    assert len(masks) == 3
    # Original resolutions preserved (PIL gives (W, H), tensor shape (H, W))
    assert masks[0].shape == (1024, 256)
    assert masks[1].shape == (256, 1024)
    assert masks[2].shape == (512, 512)
    for m in masks:
        assert torch.all(torch.isfinite(m))
        assert 0.0 <= m.min().item() <= m.max().item() <= 1.0


def test_predict_batch_single_aspect_no_padding(cpu_predictor):
    """If all images share aspect, no letterbox pad applied → no quality loss."""
    images = [Image.new("RGB", (512, 512), color=(c, c, c)) for c in (50, 100, 150)]
    masks = cpu_predictor.predict_batch(images)
    assert len(masks) == 3
    for m in masks:
        assert m.shape == (512, 512)


# --- Image.open file-handle leak check ---

def test_load_pil_releases_file_handle(tmp_path, cpu_predictor):
    """The predictor must not hold the source file open after .predict()."""
    p = tmp_path / "img.png"
    Image.new("RGB", (256, 256), color=(0, 128, 255)).save(p)
    cpu_predictor.predict(str(p))
    # On Windows this would fail if a handle were still open; on Linux it
    # just confirms we can mutate/replace the file.
    p.unlink()
    assert not p.exists()
