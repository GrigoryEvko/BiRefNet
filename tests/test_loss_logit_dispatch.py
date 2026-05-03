"""Regression tests for loss.py: BCE under autocast and StructureLoss logit dispatch.

These exercise PixLoss directly (no model needed). Verifies:

  - PixLoss runs cleanly under bf16 autocast (was a hard RuntimeError before)
  - StructureLoss receives logits, not sigmoid'd preds
  - non-logit criteria (iou, mae, ssim) still receive sigmoid'd preds
"""
from __future__ import annotations

import importlib
import sys
import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch, tmp_path):
    """Stub Config so loss.py import doesn't reach into /workspace/datasets."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Defer import until after path setup
    import config as _config

    class _Stub:
        def __init__(self):
            self.lambdas_pix_last = {
                "bce": 1.0, "iou": 1.0, "ssim": 1.0, "mae": 1.0,
                "mse": 0.0, "iou_patch": 0.0, "reg": 0.0,
                "cnt": 0.0, "structure": 1.0,
            }
            self.lambdas_cls = {"ce": 1.0}
            self.batch_size = 2

    monkeypatch.setattr(_config, "Config", lambda: _Stub())
    yield


def _fresh_pixloss():
    """Reload loss.py with the patched Config so the criterions wire correctly."""
    if "loss" in sys.modules:
        del sys.modules["loss"]
    return importlib.import_module("loss").PixLoss()


def test_pixloss_uses_bce_with_logits():
    pix = _fresh_pixloss()
    assert isinstance(pix.criterions_last["bce"], torch.nn.BCEWithLogitsLoss)


def test_pixloss_runs_under_bf16_autocast():
    """The whole point: training step must not raise RuntimeError."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for autocast")
    pix = _fresh_pixloss().cuda()
    preds = [torch.randn(2, 1, 64, 64, device="cuda", requires_grad=True) for _ in range(4)]
    gt = torch.rand(2, 1, 64, 64, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = pix(preds, gt)
    loss.backward()
    assert torch.isfinite(loss).item()


def test_structure_loss_receives_logits_not_sigmoid():
    """Hook into StructureLoss to capture what input it gets and assert it's
    the raw pred, not sigmoid(pred)."""
    pix = _fresh_pixloss()
    captured = {}
    real_forward = pix.criterions_last["structure"].forward

    def spy(self_, pred, target):
        captured["pred"] = pred
        return real_forward(pred, target)

    type(pix.criterions_last["structure"]).forward = spy
    try:
        preds = [torch.randn(2, 1, 32, 32)]
        gt = torch.rand(2, 1, 32, 32)
        pix(preds, gt)
        # If we'd silently sigmoid'd, captured['pred'] would be in (0,1).
        # Logits should leak outside (0,1) given our randn input.
        assert captured["pred"].max() > 1.0 or captured["pred"].min() < 0.0
    finally:
        type(pix.criterions_last["structure"]).forward = real_forward


def test_iou_still_receives_sigmoid_inputs():
    """Counterpart: non-logit criteria must still get sigmoid'd preds (their
    formulas assume probabilities in [0,1])."""
    pix = _fresh_pixloss()
    captured = {}
    real_forward = pix.criterions_last["iou"].forward

    def spy(self_, pred, target):
        captured["pred"] = pred
        return real_forward(pred, target)

    type(pix.criterions_last["iou"]).forward = spy
    try:
        preds = [torch.randn(1, 1, 16, 16) * 5]  # logits well outside [0,1]
        gt = torch.rand(1, 1, 16, 16)
        pix(preds, gt)
        assert captured["pred"].min() >= 0.0
        assert captured["pred"].max() <= 1.0
    finally:
        type(pix.criterions_last["iou"]).forward = real_forward
