"""End-to-end tests for the BiRefNetPredictor using a synthetic mock model.

We never load the real BiRefNet here — it requires `cv2` and `kornia` and pulls
in `Config()` which reads `train.sh`. Instead we register a tiny conv-only model
that produces a deterministic mask. This proves the predictor pipeline works:
loading, bucketing, normalization, autocast, mask resize back to original res,
RGBA composition.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from birefnet_api import BiRefNetPredictor


class MockBiRefNet(nn.Module):
    """Predicts a center-bright mask the same spatial size as the input.

    Output shape contract matches BiRefNet's inference path: returns a list of
    multiscale predictions, the last of which is the high-res logit map [B,1,H,W].
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        # Initialize so the output is mostly determined by input luminance,
        # which makes test assertions stable regardless of random init.
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0]]],
                                                  [[[0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0]]],
                                                  [[[0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0]]]]).reshape(1, 3, 3, 3))

    def forward(self, x):
        # Produce a logit map; predictor will sigmoid externally.
        logit = self.conv(x)
        return [logit, logit, logit, logit]  # multiscale list, last is final


@pytest.fixture
def cpu_predictor():
    return BiRefNetPredictor(MockBiRefNet(), device="cpu", dtype=None, max_edge=512)


def _checker_image(h: int, w: int) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[::32, :] = 255
    arr[:, ::32] = 255
    return Image.fromarray(arr)


def test_predict_returns_original_resolution(cpu_predictor):
    img = _checker_image(900, 1280)
    mask = cpu_predictor.predict(img)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (900, 1280)
    assert mask.dtype == torch.float32
    assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_predict_extreme_hr_downscales_then_upscales(cpu_predictor):
    """The user's scenario: 12K input → mask should come back at 12K."""
    img = _checker_image(8000, 12000)
    mask = cpu_predictor.predict(img, max_edge=1024)
    assert mask.shape == (8000, 12000)
    assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_predict_accepts_numpy_uint8(cpu_predictor):
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = cpu_predictor.predict(arr)
    assert mask.shape == (480, 640)


def test_predict_accepts_numpy_float(cpu_predictor):
    arr = np.random.rand(480, 640, 3).astype(np.float32)
    mask = cpu_predictor.predict(arr)
    assert mask.shape == (480, 640)


def test_predict_accepts_torch_tensor(cpu_predictor):
    t = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
    mask = cpu_predictor.predict(t)
    assert mask.shape == (480, 640)


def test_predict_accepts_path(tmp_path, cpu_predictor):
    p = tmp_path / "img.png"
    _checker_image(256, 256).save(p)
    mask = cpu_predictor.predict(str(p))
    assert mask.shape == (256, 256)


def test_predict_returns_pil_when_requested(cpu_predictor):
    img = _checker_image(400, 600)
    out = cpu_predictor.predict(img, return_pil=True)
    assert isinstance(out, Image.Image)
    assert out.mode == "L"
    assert out.size == (600, 400)  # PIL uses (W, H)


def test_predict_batch_heterogeneous_aspect(cpu_predictor):
    images = [_checker_image(400, 600), _checker_image(600, 400), _checker_image(800, 800)]
    masks = cpu_predictor.predict_batch(images)
    assert len(masks) == 3
    assert masks[0].shape == (400, 600)
    assert masks[1].shape == (600, 400)
    assert masks[2].shape == (800, 800)


def test_cutout_returns_rgba_at_original_resolution(cpu_predictor):
    img = _checker_image(720, 1280)
    out = cpu_predictor.cutout(img)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGBA"
    assert out.size == (1280, 720)
    arr = np.asarray(out)
    assert arr.shape == (720, 1280, 4)
    # alpha channel exists and has variance (mock model gives non-trivial mask)
    alpha = arr[..., 3]
    assert alpha.dtype == np.uint8


def test_cutout_with_refine_runs_at_hr(cpu_predictor):
    img = _checker_image(800, 1024)
    out = cpu_predictor.cutout(img, refine_fg=True, fg_radius=12)
    assert out.size == (1024, 800)
    assert out.mode == "RGBA"


def test_predictor_disables_grad_on_load():
    pred = BiRefNetPredictor(MockBiRefNet(), device="cpu", dtype=None)
    for p in pred.model.parameters():
        assert p.requires_grad is False


def test_predictor_does_not_import_config():
    """Sanity check: importing birefnet_api in a fresh subprocess must not
    transitively import the project's Config() (which reads train.sh and
    probes paths under SYS_HOME_DIR).
    """
    import subprocess
    import textwrap
    repo = pathlib_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(repo)!r})
        import birefnet_api  # noqa: F401
        # config must NOT have been transitively imported
        assert "config" not in sys.modules, sorted(sys.modules.keys())
    """)
    res = subprocess.run(
        ["python3", "-c", code], capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr


def test_predict_handles_grayscale_input(cpu_predictor):
    img = Image.new("L", (256, 384), color=128)
    mask = cpu_predictor.predict(img)
    assert mask.shape == (384, 256)


def test_predict_handles_rgba_input_drops_alpha(cpu_predictor):
    img = Image.new("RGBA", (256, 384), color=(120, 200, 50, 200))
    mask = cpu_predictor.predict(img)
    assert mask.shape == (384, 256)


def test_buckets_constructor_constrains_outputs():
    pred = BiRefNetPredictor(
        MockBiRefNet(),
        device="cpu",
        dtype=None,
        buckets=[(1024, 1024), (768, 1024), (1024, 768)],
    )
    img = _checker_image(2000, 1500)
    mask = pred.predict(img)
    # original resolution preserved
    assert mask.shape == (2000, 1500)
