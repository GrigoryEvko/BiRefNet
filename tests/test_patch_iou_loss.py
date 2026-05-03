"""Regression tests for PatchIoULoss vectorization + bug fix.

The original implementation iterated `range(0, target.shape[0], 64)` and
`range(0, target.shape[1], 64)` — batch and channel dims, not H/W. With
typical batches (B<64), only the top-left 64×64 tile of every prediction
was actually scored.

The vectorized rewrite tiles H and W correctly. These tests verify:
  - tiling produces a finite loss for the obvious shapes
  - identical pred/target → loss == 0 across all tiles
  - the loss respects the divisibility-via-pad path
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    import config as _config
    class _Stub:
        def __init__(self):
            self.lambdas_pix_last = {"iou_patch": 1.0}
            self.lambdas_cls = {"ce": 1.0}
            self.batch_size = 2
    monkeypatch.setattr(_config, "Config", lambda: _Stub())
    yield


def _fresh_patch_iou():
    if "loss" in sys.modules:
        del sys.modules["loss"]
    return importlib.import_module("loss").PatchIoULoss()


def test_patch_iou_zero_on_identical_inputs():
    loss_fn = _fresh_patch_iou()
    pred = torch.full((2, 1, 128, 128), 1.0)
    target = torch.full((2, 1, 128, 128), 1.0)
    out = loss_fn(pred, target).item()
    assert abs(out) < 1e-5, f"expected ~0, got {out}"


def test_patch_iou_finite_on_random_inputs():
    loss_fn = _fresh_patch_iou()
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 128, 128)
    target = torch.rand(2, 1, 128, 128)
    out = loss_fn(pred, target)
    assert torch.isfinite(out).item()
    # 2 batches × 2 × 2 tiles = 8 sub-IoUs each contributing to (1-IoU).sum().
    # With random pred/target, each sub-IoU is well below 1 → loss > 0.
    assert out.item() > 0


def test_patch_iou_pads_non_divisible_h():
    """Spatial dims not multiple of 64 → pad to next multiple."""
    loss_fn = _fresh_patch_iou()
    pred = torch.rand(1, 1, 70, 128)  # 70 isn't divisible by 64
    target = torch.rand(1, 1, 70, 128)
    # Should not raise, should return a finite scalar.
    out = loss_fn(pred, target)
    assert torch.isfinite(out).item()


def test_patch_iou_actually_iterates_over_h_w():
    """Sanity: zeroing the BOTTOM-RIGHT tile only must change the loss
    relative to a pred that matches target everywhere. The old buggy code
    ignored the bottom-right tile entirely — only top-left was scored.
    """
    loss_fn = _fresh_patch_iou()
    target = torch.full((1, 1, 128, 128), 1.0)
    pred_match = target.clone()
    pred_corrupted = target.clone()
    pred_corrupted[:, :, 64:, 64:] = 0.0  # zero out bottom-right tile
    a = loss_fn(pred_match, target).item()
    b = loss_fn(pred_corrupted, target).item()
    assert b > a + 0.5, (
        f"corrupting the bottom-right tile must raise the loss; got match={a} corrupted={b}"
    )
