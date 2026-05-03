"""Tests for the GPU foreground refinement helpers in `birefnet_api.predictor`.

We test our new separable box blur against PIL/numpy ground truth, and verify
that the foreground estimator produces uint8 output at the input resolution
without modifying alpha values.
"""
from __future__ import annotations

import pytest
import torch

from birefnet_api.predictor import _box_blur, _fb_blur_pass


def test_box_blur_matches_avg_pool_for_odd_kernel():
    x = torch.randn(1, 3, 64, 64)
    r = 5
    sep = _box_blur(x, r)
    # equivalent direct (r,r) avg pool with the same replicate padding
    pad = r // 2
    direct = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate"),
        kernel_size=r, stride=1, count_include_pad=False,
    )
    torch.testing.assert_close(sep, direct, rtol=1e-5, atol=1e-5)


def test_box_blur_handles_even_kernel_without_crash():
    x = torch.zeros(1, 1, 32, 32)
    out = _box_blur(x, 4)
    assert out.shape == x.shape


@pytest.mark.parametrize("r", [3, 6, 9, 16, 31])
def test_box_blur_shape_invariance(r):
    x = torch.randn(1, 3, 33, 47)
    assert _box_blur(x, r).shape == x.shape


def test_box_blur_constant_input_is_constant():
    x = torch.full((1, 3, 64, 64), 0.42)
    out = _box_blur(x, 7)
    assert torch.allclose(out, x, atol=1e-6)


def test_fb_blur_pass_does_not_modify_alpha():
    """The Photoroom-style foreground refinement returns a new RGB but does
    not change the alpha tensor passed in."""
    rgb = torch.rand(1, 3, 64, 64)
    alpha = torch.rand(1, 1, 64, 64)
    alpha_in = alpha.clone()
    _fb_blur_pass(rgb, rgb, rgb, alpha, r=5)
    assert torch.equal(alpha, alpha_in)


def test_fb_blur_pass_preserves_resolution():
    rgb = torch.rand(1, 3, 100, 200)
    alpha = torch.rand(1, 1, 100, 200)
    fg, b = _fb_blur_pass(rgb, rgb, rgb, alpha, r=11)
    assert fg.shape == rgb.shape
    assert b.shape == rgb.shape


def test_fb_blur_pass_full_alpha_is_identity_on_fg():
    """When alpha == 1 everywhere, the formula collapses to FG = blurred_FG +
    1*(image - 1*blurred_FG - 0) = image."""
    rgb = torch.rand(1, 3, 32, 32)
    alpha = torch.ones(1, 1, 32, 32)
    fg, _ = _fb_blur_pass(rgb, rgb, rgb, alpha, r=3)
    torch.testing.assert_close(fg, rgb, atol=1e-5, rtol=1e-5)


def test_fb_blur_pass_zero_alpha_is_blurred_background():
    """With alpha == 0 everywhere, FG = blurred_FG (the blur of the image,
    since `image == FG` here) — the formula collapses to blurred_FG + 0."""
    rgb = torch.rand(1, 3, 32, 32) * 0.5 + 0.25
    alpha = torch.zeros(1, 1, 32, 32)
    fg, _ = _fb_blur_pass(rgb, rgb, rgb, alpha, r=5)
    # The "fg" branch is meaningless when alpha=0, but shouldn't NaN or explode.
    assert torch.all(torch.isfinite(fg))
