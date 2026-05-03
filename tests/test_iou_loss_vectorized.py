"""Regression test for IoULoss vectorization.

The original implementation looped over the batch in Python, summing
(1 - IoU_i). The vectorized version computes per-batch IoU in one pass
and sums; tests verify numerical equivalence across non-trivial batches.
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
    """loss.py imports config at module level; patch to a minimal stub."""
    import config as _config
    class _Stub:
        def __init__(self):
            self.lambdas_pix_last = {"iou": 1.0}
            self.lambdas_cls = {"ce": 1.0}
            self.batch_size = 2
    monkeypatch.setattr(_config, "Config", lambda: _Stub())
    yield


def _fresh_iou():
    if "loss" in sys.modules:
        del sys.modules["loss"]
    return importlib.import_module("loss").IoULoss()


def _ref_iou_loop(pred, target):
    """Original Python-loop implementation, for equivalence comparison."""
    b = pred.shape[0]
    IoU = 0.0
    for i in range(b):
        inter = (target[i, :, :, :] * pred[i, :, :, :]).sum()
        union = target[i, :, :, :].sum() + pred[i, :, :, :].sum() - inter
        IoU = IoU + (1 - inter / union)
    return IoU


def test_iou_matches_loop_on_random_inputs():
    iou = _fresh_iou()
    torch.manual_seed(0)
    for _ in range(5):
        pred = torch.rand(4, 1, 32, 32)
        target = torch.rand(4, 1, 32, 32)
        a = iou(pred, target).item()
        b = _ref_iou_loop(pred, target).item()
        assert abs(a - b) < 1e-5, (a, b)


def test_iou_handles_batch_size_1():
    iou = _fresh_iou()
    pred = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    a = iou(pred, target).item()
    b = _ref_iou_loop(pred, target).item()
    assert abs(a - b) < 1e-5


def test_iou_grad_flows_through_pred():
    iou = _fresh_iou()
    pred = torch.rand(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 8, 8)
    iou(pred, target).backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
