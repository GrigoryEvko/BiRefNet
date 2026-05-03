"""End-to-end integration tests with the real BiRefNet model.

These tests instantiate the actual `models.birefnet.BiRefNet` (no pretrained
weights — we just exercise the graph for shape/contract guarantees), wrap it
in our `BiRefNetPredictor`, and run the full pipeline:
  - arbitrary-resolution input → bucket → real model forward → mask upsample
  - 12K → bucket → mask back at 12K
  - state-dict round-trip with compile/DDP prefixes
  - autocast bf16 path (skipped if no CUDA)

These are slow-ish (a few seconds each) but they exercise the parts that the
unit tests can only mock. Skipped if cv2/kornia aren't installed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

cv2 = pytest.importorskip("cv2")
kornia = pytest.importorskip("kornia")

# Repo root on sys.path so we can import the project's models.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def real_predictor():
    """Build a fresh BiRefNet (no pretrained backbone) at the smallest backbone
    that exists in the repo: swin_v1_l with random init at 1024 native, but
    we use max_edge=256 to keep this test fast."""
    from birefnet_api import BiRefNetPredictor

    # We need to override the project's Config() because constructing a real
    # model triggers it, and Config.__init__ probes /workspace/datasets and
    # train.sh paths. Patch the offending lines lazily.
    import config as _config
    _orig_init = _config.Config.__init__

    def _safe_init(self):
        try:
            _orig_init(self)
        except FileNotFoundError:
            # The dataset listing failed; everything else is set. We can
            # ignore and proceed for the model graph.
            pass
    _config.Config.__init__ = _safe_init
    try:
        from models.birefnet import BiRefNet
        model = BiRefNet(bb_pretrained=False)
    finally:
        _config.Config.__init__ = _orig_init

    return BiRefNetPredictor(model, device="cpu", dtype=None, max_edge=256)


def _gradient_image(h: int, w: int) -> Image.Image:
    """Smooth gradient image — gives the predictor a non-trivial input even
    without pretrained weights, so we can sanity-check the mask is finite."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    arr = np.stack([
        (xx * 255).astype(np.uint8),
        (yy * 255).astype(np.uint8),
        ((1 - xx) * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)


def test_predict_shape_round_trips_at_arbitrary_resolution(real_predictor):
    img = _gradient_image(384, 512)
    mask = real_predictor.predict(img)
    assert mask.shape == (384, 512)
    assert mask.dtype == torch.float32
    assert torch.all(torch.isfinite(mask))
    assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_predict_returns_mask_at_arbitrary_hr(real_predictor):
    """The user's scenario: an HR source goes in, mask comes back at the
    same resolution. We use 4096×3000 as the proxy for 'arbitrary HR' to
    keep the test tractable on a laptop; the bucketing logic is identical
    at 12K (just allocates more memory).
    """
    h, w = 3000, 4096
    img = _gradient_image(h, w)
    mask = real_predictor.predict(img)
    assert mask.shape == (h, w)
    assert torch.all(torch.isfinite(mask))
    assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_predict_returns_mask_at_true_12k_via_bucket():
    """Skip-able: full 12K test. Use a synthetic black-and-white image to
    minimize memory pressure, and a tiny bucket. Skipped by default because
    even with bucket=128, allocating a 12000x8000 mask in fp32 is ~384MB.
    Set BIREFNET_TEST_12K=1 to enable.
    """
    if not os.environ.get("BIREFNET_TEST_12K"):
        pytest.skip("set BIREFNET_TEST_12K=1 to run the full 12K test")
    from birefnet_api import BiRefNetPredictor
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

    pred = BiRefNetPredictor(model, device="cpu", dtype=None, max_edge=128)
    img = _gradient_image(8000, 12000)
    mask = pred.predict(img)
    assert mask.shape == (8000, 12000)


def test_cutout_returns_rgba_at_original_resolution(real_predictor):
    img = _gradient_image(360, 480)
    rgba = real_predictor.cutout(img)
    assert rgba.size == (480, 360)  # PIL is (W, H)
    assert rgba.mode == "RGBA"
    arr = np.asarray(rgba)
    assert arr.shape == (360, 480, 4)


def test_state_dict_round_trip_with_compile_ddp_prefixes(tmp_path, real_predictor):
    """Save the state dict with synthetic compile/DDP prefixes, reload via
    check_state_dict, ensure the model still loads cleanly."""
    from utils import check_state_dict

    sd = real_predictor.model.state_dict()
    # Synthesize a wrapped state dict like accelerate + torch.compile produces:
    wrapped = {f"_orig_mod.module.{k}": v for k, v in sd.items()}
    # Round-trip
    out = check_state_dict(wrapped)
    # All original keys must come back intact
    assert set(out.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(out[k], sd[k])
    # Reloading into the model must not raise
    real_predictor.model.load_state_dict(out)


def test_predict_batch_real_model_heterogeneous(real_predictor):
    images = [_gradient_image(192, 256), _gradient_image(256, 192), _gradient_image(160, 320)]
    masks = real_predictor.predict_batch(images)
    assert len(masks) == 3
    assert masks[0].shape == (192, 256)
    assert masks[1].shape == (256, 192)
    assert masks[2].shape == (160, 320)
    for m in masks:
        assert torch.all(torch.isfinite(m))
        assert 0.0 <= m.min().item() <= m.max().item() <= 1.0


def test_predict_with_bf16_autocast_on_cuda(real_predictor):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Some torchvision wheels don't have a CUDA build of deform_conv2d.
    # The bf16 fp32-cast in our DeformableConv2d only helps if a CUDA kernel
    # exists at all. Probe the op once and skip the test if it isn't there.
    try:
        from torchvision.ops import deform_conv2d
        x = torch.zeros(1, 1, 4, 4, device="cuda")
        offset = torch.zeros(1, 2 * 1 * 1, 4, 4, device="cuda")
        weight = torch.zeros(1, 1, 1, 1, device="cuda")
        deform_conv2d(x, offset, weight)
    except NotImplementedError as e:
        if "CUDA" in str(e):
            pytest.skip("torchvision lacks a CUDA deform_conv2d kernel")
        raise

    from birefnet_api import BiRefNetPredictor
    pred = BiRefNetPredictor(real_predictor.model, device="cuda", dtype="bf16", max_edge=256)
    img = _gradient_image(256, 256)
    mask = pred.predict(img)
    assert mask.shape == (256, 256)
    assert torch.all(torch.isfinite(mask))
