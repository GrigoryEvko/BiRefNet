"""CUDA bf16 smoke test — runs only on CUDA hardware.

Verifies the bf16 autocast path actually works end-to-end through the
real model: forward, sigmoid, upsample, mask. Skipped on CPU-only CI
(can't be exercised without GPU). Run on the user's L40S before
production.

Set BIREFNET_CUDA_BF16_TEST=1 to opt into running this even if other
gating skips it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")
pytest.importorskip("cv2")


def _have_cuda_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    # bf16 needs Ampere (compute ≥ 8.0) or newer for tensor-core kernels.
    # Older GPUs accept bf16 ops but fall through to fp32 emulation, which
    # still validates correctness — but we'd like at least one capability check.
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


@pytest.fixture(scope="module")
def real_predictor_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not _have_cuda_bf16() and not os.environ.get("BIREFNET_CUDA_BF16_TEST"):
        pytest.skip(
            "GPU compute capability < 8.0 — bf16 paths fall back to fp32 emulation. "
            "Set BIREFNET_CUDA_BF16_TEST=1 to run anyway."
        )
    # Probe deform_conv2d on CUDA — if torchvision lacks the kernel we can't
    # exercise the real model graph.
    try:
        from torchvision.ops import deform_conv2d
        x = torch.zeros(1, 1, 4, 4, device="cuda")
        offset = torch.zeros(1, 2, 4, 4, device="cuda")
        weight = torch.zeros(1, 1, 1, 1, device="cuda")
        deform_conv2d(x, offset, weight)
    except (NotImplementedError, RuntimeError) as e:
        pytest.skip(f"torchvision lacks CUDA deform_conv2d kernel: {e}")

    # Build the real model under a stubbed Config (so we don't need the
    # /workspace/datasets layout).
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
        from birefnet_api import BiRefNetPredictor
        model = BiRefNet(bb_pretrained=False)
        pred = BiRefNetPredictor(model, device="cuda", dtype="bf16", max_edge=384)
    finally:
        _config.Config.__init__ = _orig
    return pred


def _gradient_image(h: int, w: int):
    from PIL import Image
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    arr = np.stack([
        (xx * 255).astype(np.uint8),
        (yy * 255).astype(np.uint8),
        ((1 - xx) * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)


def test_cuda_bf16_predict_returns_finite_mask(real_predictor_cuda):
    img = _gradient_image(384, 512)
    mask = real_predictor_cuda.predict(img)
    assert mask.shape == (384, 512)
    assert mask.dtype == torch.float32  # contract: cast to fp32 at boundary
    assert torch.all(torch.isfinite(mask))
    assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_cuda_bf16_batch_finite_across_aspects(real_predictor_cuda):
    """Heterogeneous batch through bf16 autocast: letterbox + cast + crop."""
    images = [_gradient_image(192, 256), _gradient_image(256, 192), _gradient_image(160, 320)]
    masks = real_predictor_cuda.predict_batch(images)
    assert len(masks) == 3
    for m in masks:
        assert m.dtype == torch.float32
        assert torch.all(torch.isfinite(m))


def test_cuda_bf16_warmup_then_predict(real_predictor_cuda):
    """Warmup primes compile/autotune state; subsequent predict still works."""
    real_predictor_cuda.warmup(shapes=[(384, 384)])
    img = _gradient_image(384, 384)
    mask = real_predictor_cuda.predict(img)
    assert mask.shape == (384, 384)
    assert torch.all(torch.isfinite(mask))


def test_cuda_bf16_no_grad_state_leaks(real_predictor_cuda):
    """Running predict under inference_mode must NOT enable grads on params."""
    img = _gradient_image(192, 256)
    real_predictor_cuda.predict(img)
    assert not any(p.requires_grad for p in real_predictor_cuda.model.parameters()), (
        "predict() side-effect: model parameters got requires_grad=True"
    )
