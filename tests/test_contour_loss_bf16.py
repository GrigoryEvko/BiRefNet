"""Regression test: ContourLoss must produce finite gradients under bf16.

The sqrt(delta_pred + 1e-8) is numerically fragile: in bf16, 1e-8 has no
representable significand and underflows to 0. sqrt(0).backward() emits
+inf, poisoning the optimizer state. ContourLoss now upcasts to fp32
internally regardless of autocast.
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
            self.lambdas_pix_last = {"cnt": 1.0}
            self.lambdas_cls = {"ce": 1.0}
            self.batch_size = 2
    monkeypatch.setattr(_config, "Config", lambda: _Stub())
    yield


def _fresh_contour():
    if "loss" in sys.modules:
        del sys.modules["loss"]
    return importlib.import_module("loss").ContourLoss()


def test_contour_loss_finite_under_bf16_input():
    """Pass bf16 inputs directly (no autocast) → loss + gradients still finite."""
    loss_fn = _fresh_contour()
    pred = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(pred.grad).all().item()


def test_contour_loss_zero_input_grad_finite():
    """Stress: pred is identically 0 → delta_pred is 0 → sqrt(0+eps).backward
    must not produce inf, in fp32 or bf16."""
    loss_fn = _fresh_contour()
    for dtype in (torch.float32, torch.bfloat16):
        pred = torch.zeros(1, 1, 8, 8, dtype=dtype, requires_grad=True)
        target = torch.zeros(1, 1, 8, 8, dtype=dtype)
        loss = loss_fn(pred, target)
        loss.backward()
        assert torch.isfinite(pred.grad).all().item(), f"non-finite grad in {dtype}"
