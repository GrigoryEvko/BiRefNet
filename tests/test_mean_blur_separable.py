"""Regression test: mean_blur is now separable but must produce the same
numerical result as the pre-existing (k×k) avg_pool2d implementation.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


def _ref_mean_blur(x, kernel_size):
    """Single-pass (k×k) reference, count_include_pad=True to match the
    separable implementation."""
    if kernel_size % 2 == 0:
        pad_l = kernel_size // 2 - 1
        pad_r = kernel_size // 2
        pad_t = kernel_size // 2 - 1
        pad_b = kernel_size // 2
    else:
        pad_l = pad_r = pad_t = pad_b = kernel_size // 2
    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
    return torch.nn.functional.avg_pool2d(
        x_padded, kernel_size=(kernel_size, kernel_size), stride=1, count_include_pad=True
    )


@pytest.mark.parametrize("k", [3, 5, 7, 9, 16, 31, 32, 90])
def test_mean_blur_matches_reference(k):
    from image_proc import mean_blur
    torch.manual_seed(0)
    # Input larger than k so we test interior averaging rather than the
    # "kernel >= input" boundary case.
    x = torch.rand(1, 3, 128, 128)
    a = mean_blur(x, k)
    b = _ref_mean_blur(x, k)
    # Tolerance scales with k: separable does k+k summands in different
    # order than single-pass does k*k summands; fp32 reordering accumulates
    # ~k * ulp(0.5) ≈ 6e-7 * k. atol=1e-4 covers k up to ~150.
    assert torch.allclose(a, b, atol=1e-4), f"separable diverges at k={k}"


def test_mean_blur_shape_preserved():
    from image_proc import mean_blur
    x = torch.rand(2, 1, 32, 32)
    out = mean_blur(x, 7)
    assert out.shape == x.shape
