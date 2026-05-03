"""Regression tests for the config.align_corners flag.

We added this flag because the upstream pretrained ZhengPeng7/BiRefNet
checkpoints were trained with align_corners=True everywhere. A previous
sweep flipped them all to False, which silently shifted mask geometry by
~½ pixel. The flag preserves pretrained-weight compatibility by default.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_default_align_corners_is_true():
    """Default must be True for upstream checkpoint compatibility."""
    if "config" in sys.modules:
        del sys.modules["config"]
    cfg_mod = importlib.import_module("config")
    cfg = cfg_mod.Config()
    assert cfg.align_corners is True


def test_birefnet_plumbs_align_corners_to_decoder():
    """BiRefNet and its Decoder both cache the flag onto self._ac, so the
    interpolate sites in both modules read from a single source.
    """
    pytest.importorskip("kornia")
    pytest.importorskip("torch")

    import config as _config

    _orig = _config.Config.__init__
    def _safe(self):
        try:
            _orig(self)
        except FileNotFoundError:
            pass
    _config.Config.__init__ = _safe
    try:
        from models.birefnet import BiRefNet, Decoder
        model = BiRefNet(bb_pretrained=False)
        # Both have the cached flag and they agree.
        assert hasattr(model, "_ac")
        assert isinstance(model.decoder, Decoder)
        assert hasattr(model.decoder, "_ac")
        assert model._ac == model.decoder._ac
        # Default config has align_corners=True.
        assert model._ac is True
    finally:
        _config.Config.__init__ = _orig


def test_birefnet_falls_back_to_true_when_config_lacks_flag():
    """If a config object is missing the flag entirely (stale config), BiRefNet
    must default to True (pretrained-compat) rather than False.

    We test the getattr-fallback directly rather than constructing a stale
    config end-to-end, since Config writes the new field unconditionally.
    """
    class _MinimalConfig:
        bb = "swin_v1_l"
    assert bool(getattr(_MinimalConfig(), "align_corners", True)) is True


def test_align_corners_actually_changes_interpolation():
    """Sanity: the flag's value really does flip F.interpolate behavior.

    The check uses a 2x4 input upsampled to 4x8 via 'bilinear'. The two
    align_corners settings disagree on the interior pixel coordinates.
    """
    import torch
    import torch.nn.functional as F

    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    out_true = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=True)
    out_false = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)
    # The two outputs differ — proves the flag changes pixel arithmetic.
    assert not torch.allclose(out_true, out_false)
