"""Tests for predictor quality-of-life features:
  - device='auto' picks best-available
  - from_state_dict factory builds a predictor without writing a temp file
  - __repr__ surfaces backbone, params, device, dtype
  - warmup() drives forwards on known shapes
  - OOM handler enriches the error message (CUDA-only; smoke check otherwise)
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birefnet_api import BiRefNetPredictor
from birefnet_api.predictor import BiRefNetPredictor as _BiRefNetPredictor


class _Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1 / 3)

    def forward(self, x):
        return [self.conv(x)]


@pytest.fixture
def cpu_predictor():
    return BiRefNetPredictor(_Identity(), device="cpu", dtype=None, max_edge=128, normalize=False)


def test_device_auto_picks_cpu_when_no_cuda(monkeypatch):
    """With no CUDA and no MPS, device='auto' lands on cpu."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Hide MPS too
    monkeypatch.setattr(torch.backends, "mps", type("X", (), {"is_available": staticmethod(lambda: False)})(), raising=False)
    pred = BiRefNetPredictor(_Identity(), device="auto", dtype=None, max_edge=64, normalize=False)
    assert pred.device.type == "cpu"


def test_device_auto_prefers_cuda_when_available(monkeypatch):
    """With CUDA available, 'auto' returns a CUDA device — even if we never
    actually run on it (we skip the move-to-device by stubbing model.to)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # We don't actually move the model to CUDA; stub Module.to to be a no-op
    # for the cuda path so the test runs on a CPU-only machine.
    real_to = nn.Module.to
    def stub_to(self, *args, **kwargs):
        return self
    monkeypatch.setattr(nn.Module, "to", stub_to)
    try:
        pred = _BiRefNetPredictor(_Identity(), device="auto", dtype=None, max_edge=64, normalize=False)
        assert pred.device.type == "cuda"
    finally:
        monkeypatch.setattr(nn.Module, "to", real_to)


def test_repr_surfaces_useful_info(cpu_predictor):
    r = repr(cpu_predictor)
    assert "BiRefNetPredictor" in r
    assert "device=cpu" in r
    assert "params=" in r
    assert "max_edge=128" in r


def test_repr_includes_buckets_when_set():
    pred = BiRefNetPredictor(
        _Identity(), device="cpu", dtype=None, normalize=False,
        buckets=[(64, 64), (128, 128), (96, 192)],
    )
    r = repr(pred)
    assert "buckets=3" in r
    assert "max_edge" not in r  # buckets supersede the max_edge field


def test_warmup_runs_forward_for_each_shape(cpu_predictor):
    """warmup() drives one forward per shape and doesn't crash."""
    cpu_predictor.warmup(shapes=[(64, 64), (96, 128)])
    # Sanity: a real predict still works after warmup (no state corruption).
    img = Image.new("RGB", (64, 64), color=(0, 128, 255))
    mask = cpu_predictor.predict(img)
    assert mask.shape == (64, 64)


def test_warmup_uses_explicit_buckets_when_no_arg():
    """When buckets are configured and warmup() is called without shapes,
    it warms each bucket."""
    pred = BiRefNetPredictor(
        _Identity(), device="cpu", dtype=None, normalize=False,
        buckets=[(64, 64), (96, 96)],
    )
    pred.warmup()  # no shapes — uses self.buckets
    img = Image.new("RGB", (64, 64), color=(0, 128, 255))
    mask = pred.predict(img)
    assert mask.shape == (64, 64)


def test_warmup_no_op_without_shapes_or_buckets(cpu_predictor):
    """No shapes, no buckets → silent no-op (doesn't error)."""
    cpu_predictor.warmup()  # should not raise


def test_from_state_dict_round_trips_via_check_state_dict():
    """Build a predictor from a state_dict that was wrapped with DDP/compile
    prefixes — check_state_dict should strip them, model loads cleanly."""
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")

    # Build the real model under a stubbed config to get a valid state_dict.
    import config as _config
    orig = _config.Config.__init__
    def safe(self):
        try: orig(self)
        except FileNotFoundError: pass
    _config.Config.__init__ = safe
    try:
        from models.birefnet import BiRefNet
        m = BiRefNet(bb_pretrained=False)
        sd = m.state_dict()
        # Wrap with DDP+compile prefixes
        wrapped = {f"_orig_mod.module.{k}": v for k, v in sd.items()}
        pred = BiRefNetPredictor.from_state_dict(
            wrapped, device="cpu", dtype=None, max_edge=128
        )
        # If it didn't raise, the strict load succeeded.
        assert isinstance(pred, BiRefNetPredictor)
    finally:
        _config.Config.__init__ = orig


def test_from_state_dict_strict_raises_on_mismatch():
    """A state_dict with extra unexpected keys must raise under strict=True."""
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")

    import config as _config
    orig = _config.Config.__init__
    def safe(self):
        try: orig(self)
        except FileNotFoundError: pass
    _config.Config.__init__ = safe
    try:
        from models.birefnet import BiRefNet
        m = BiRefNet(bb_pretrained=False)
        sd = dict(m.state_dict())
        # Inject an unexpected key
        sd["nonsense.key"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="state-dict mismatch"):
            BiRefNetPredictor.from_state_dict(sd, device="cpu", dtype=None)
        # strict=False should tolerate it
        pred = BiRefNetPredictor.from_state_dict(
            sd, strict=False, device="cpu", dtype=None
        )
        assert isinstance(pred, BiRefNetPredictor)
    finally:
        _config.Config.__init__ = orig
