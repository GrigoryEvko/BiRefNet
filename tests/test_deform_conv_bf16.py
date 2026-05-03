"""Regression test: DeformableConv2d must work under bf16 autocast.

torchvision.ops.deform_conv2d has no bf16 CUDA kernel; the BiRefNet decoder's
ASPPDeformable branch is the default config (`dec_att='ASPPDeformable'`), and
the default training mixed_precision is 'bf16'. Without our fp32 fallback,
every forward in the decoder raised NotImplementedError.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def deform_conv():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.modules.deform_conv import DeformableConv2d
    return DeformableConv2d


def test_deform_conv_runs_on_fp32_cpu(deform_conv):
    m = deform_conv(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 16, 16)
    y = m(x)
    assert y.shape == (1, 8, 16, 16)
    assert y.dtype == torch.float32


def test_deform_conv_handles_bf16_input_directly(deform_conv):
    """If a caller hands us a bf16 tensor (e.g. inside torch.amp.autocast),
    DeformableConv2d must transparently cast for the underlying torchvision
    op (which has no bf16 kernel) and return bf16 output."""
    m = deform_conv(3, 8, kernel_size=3, padding=1).bfloat16()
    x = torch.randn(1, 3, 16, 16, dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == (1, 8, 16, 16)
    assert y.dtype == torch.bfloat16
    assert torch.all(torch.isfinite(y))


def test_deform_conv_runs_under_cuda_bf16_autocast(deform_conv):
    """The full prod scenario — only runs if torchvision has a CUDA build."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    m = deform_conv(3, 8, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 3, 16, 16, device="cuda")
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
    except NotImplementedError as e:
        if "CUDA" in str(e):
            pytest.skip("installed torchvision lacks CUDA deform_conv2d kernel")
        raise
    assert y.shape == (1, 8, 16, 16)
    assert y.dtype == torch.bfloat16
    assert torch.all(torch.isfinite(y))
