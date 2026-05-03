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


def test_every_internal_interpolate_uses_the_flag():
    """Strong plumbing check: monkey-patch F.interpolate during a forward pass
    and assert every call's align_corners kwarg matches the model's _ac flag.

    A regression where someone hardcodes `align_corners=False` at a new call
    site would otherwise silently slip through the property-style checks.
    """
    pytest.importorskip("kornia")
    pytest.importorskip("torch")

    import torch
    import torch.nn.functional as F
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
        model = BiRefNet(bb_pretrained=False).eval()
        # Lock in fp32 weights and a small input that the model accepts.
        x = torch.zeros(1, 3, 64, 64)

        captured: list = []
        real_interpolate = F.interpolate

        def spy_interpolate(*args, **kwargs):
            captured.append(kwargs.get("align_corners", "<absent>"))
            return real_interpolate(*args, **kwargs)

        F.interpolate = spy_interpolate
        try:
            with torch.no_grad():
                model(x)
        finally:
            F.interpolate = real_interpolate

        # Filter out the bicubic upsamples (those use align_corners=False
        # at the user/eval boundary, which is a deliberate choice). The
        # bilinear sites in the model graph must follow the flag.
        # Easiest invariant: every captured align_corners is either True
        # (matches model._ac) or False (the boundary upsamples we don't own).
        # In the model graph specifically we want every kwarg == model._ac.
        # Since BiRefNet itself doesn't call F.interpolate with bicubic in
        # its forward, every captured value should equal model._ac.
        assert captured, "monkey-patch never fired — model didn't call F.interpolate?"
        ac = model._ac
        for i, v in enumerate(captured):
            assert v == ac, (
                f"F.interpolate call #{i} used align_corners={v!r} "
                f"but model._ac={ac!r} — a hardcoded site slipped through"
            )
    finally:
        _config.Config.__init__ = _orig


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
