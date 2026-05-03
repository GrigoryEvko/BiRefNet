"""Regression test for image2patches non-divisible spatial dims.

The previous implementation crashed when image.shape[-2] % grid_h != 0
(or the same on the W axis). einops rearrange requires the dimension
be divisible by the rearrange factor; we now pad bottom-right with
replicate to the next multiple before rearrange.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
einops = pytest.importorskip("einops")


def test_image2patches_handles_non_divisible_h():
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 510, 512)  # H=510 not divisible by 30 grid → would crash
    patch_ref = torch.zeros(1, 1, 17, 17)  # grid_h = 510 // 17 = 30, but 510 % 30 = 0
    out = image2patches(image, patch_ref=patch_ref,
                        transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
    # Just verify it didn't crash; the size depends on the rearrange.
    assert out.ndim == 4


def test_image2patches_explicit_grid_with_remainder():
    """Pass a grid that doesn't divide evenly — image2patches must pad."""
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 17, 17)  # neither H nor W divisible by 4
    out = image2patches(image, grid_h=4, grid_w=4,
                        transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
    # After replicate-pad to 20×20 then rearrange to (b 3*16) 5 5
    assert out.shape == (1, 3 * 16, 5, 5)


def test_image2patches_no_pad_when_divisible():
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 16, 16)
    out = image2patches(image, grid_h=4, grid_w=4,
                        transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
    assert out.shape == (1, 3 * 16, 4, 4)


def test_image2patches_handles_image_smaller_than_ref():
    """Edge case: image is smaller than patch_ref → grid would be 0; max(1, ...) clamps."""
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 4, 4)
    patch_ref = torch.zeros(1, 1, 8, 8)  # ref bigger than image
    out = image2patches(image, patch_ref=patch_ref,
                        transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
    # grid_h = max(1, 4//8) = 1; rearrange becomes a no-op shape-wise.
    assert out.ndim == 4


def test_image2patches_default_transformation_with_remainder():
    """The function's default transformation 'b c (hg h) (wg w) -> (b hg wg) c h w'
    must also pad correctly. Different output layout (batched-along-grid
    instead of channel-along-grid), but the same divisibility constraint.
    """
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 17, 17)
    out = image2patches(image, grid_h=4, grid_w=4)  # uses default transformation
    # After replicate-pad to 20×20, default transformation gives
    # (b * hg * wg) c h w = (1*4*4, 3, 5, 5) = (16, 3, 5, 5).
    assert out.shape == (16, 3, 5, 5)


def test_image2patches_default_transformation_no_pad_when_divisible():
    from models.birefnet import image2patches
    image = torch.zeros(1, 3, 16, 16)
    out = image2patches(image, grid_h=4, grid_w=4)  # default transformation
    assert out.shape == (16, 3, 4, 4)
