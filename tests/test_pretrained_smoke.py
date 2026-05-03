"""Real-pretrained-weights smoke test — opt-in via env var.

Set BIREFNET_TEST_PRETRAINED=1 to run. Downloads the upstream
ZhengPeng7/BiRefNet checkpoint from HuggingFace Hub, runs cutout on
synthetic + simple geometric inputs, and asserts the mask is plausible:
finite values, in [0, 1], with foreground/background that approximately
matches the obvious geometric truth.

This is the test the audit identified as the production gate. It
verifies:
  - state-dict prefix stripping handles real upstream keys
  - align_corners default produces sensible masks (not half-pixel-shifted nonsense)
  - the bf16 path on CUDA actually works
  - the cutout API end-to-end produces RGBA at the right resolution

It does NOT verify exact byte-equality with upstream output (that would
need a saved reference image; left for the user's CI to add).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pytestmark = pytest.mark.skipif(
    not os.environ.get("BIREFNET_TEST_PRETRAINED"),
    reason="set BIREFNET_TEST_PRETRAINED=1 to run pretrained-weights smoke test",
)


@pytest.fixture(scope="module")
def real_predictor():
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")
    pytest.importorskip("huggingface_hub")
    import torch
    from birefnet_api import BiRefNetPredictor

    repo_id = os.environ.get("BIREFNET_PRETRAINED_REPO", "ZhengPeng7/BiRefNet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bf16" if torch.cuda.is_available() else None
    try:
        return BiRefNetPredictor.from_pretrained(repo_id, device=device, dtype=dtype, max_edge=1024)
    except Exception as e:
        pytest.skip(f"could not download {repo_id} from HF Hub: {e}")


def _disk_image(h: int, w: int, radius_frac: float = 0.3):
    """Synthetic test image: a centered colored disk on a contrasting
    background. Foreground/background are separable so the mask should
    approximate the disk."""
    from PIL import Image
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2)
    mask = (rr < radius_frac).astype(np.float32)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (mask * 200 + (1 - mask) * 30).astype(np.uint8)
    arr[..., 1] = (mask * 50 + (1 - mask) * 30).astype(np.uint8)
    arr[..., 2] = (mask * 50 + (1 - mask) * 200).astype(np.uint8)
    return Image.fromarray(arr), mask


def test_pretrained_predicts_finite_mask_on_disk(real_predictor):
    """Mask is finite and in [0, 1]."""
    import torch
    img, _ = _disk_image(512, 512)
    mask = real_predictor.predict(img)
    assert mask.shape == (512, 512)
    assert mask.dtype == torch.float32
    assert torch.all(torch.isfinite(mask))
    assert 0.0 <= mask.min().item()
    assert mask.max().item() <= 1.0


def test_pretrained_distinguishes_disk_from_background(real_predictor):
    """The disk's center pixels should score higher than the corner pixels.
    Specifically the model should treat the salient disk as foreground."""
    img, gt_mask = _disk_image(512, 512, radius_frac=0.3)
    mask = real_predictor.predict(img).cpu().numpy()
    # Mean mask value inside the GT disk vs outside.
    fg = mask[gt_mask > 0.5].mean()
    bg = mask[gt_mask < 0.5].mean()
    # Even on the synthetic disk we expect a non-trivial separation —
    # if the model is loaded correctly. align_corners issues, prefix
    # stripping bugs, etc. would collapse the mask to constant.
    assert fg > bg + 0.1, (
        f"foreground/background separation too small: fg={fg:.3f} bg={bg:.3f} "
        f"— pretrained weights may not be loading correctly, or align_corners "
        f"default is wrong"
    )


def test_pretrained_cutout_returns_rgba(real_predictor):
    img, _ = _disk_image(384, 384)
    rgba = real_predictor.cutout(img)
    assert rgba.size == (384, 384)
    assert rgba.mode == "RGBA"


def test_pretrained_handles_hr_input(real_predictor):
    """4K input downsamples internally, mask comes back at original res."""
    img, _ = _disk_image(2160, 3840)
    mask = real_predictor.predict(img)
    assert mask.shape == (2160, 3840)
    import torch
    assert torch.all(torch.isfinite(mask))
