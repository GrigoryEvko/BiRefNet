"""Regression tests for the third-pass audit findings.

Each test pins one specific bug or correctness invariant the elite-bug-hunter
identified, so we don't regress in a future refactor.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birefnet_api import BiRefNetPredictor


class _Identity(nn.Module):
    """1x1 conv that averages RGB; produces a deterministic mask whose
    spatial structure mirrors input intensity. Useful for aspect tests."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1 / 3)

    def forward(self, x):
        return [self.conv(x)]


# --- MISS-1 / PERF-5: predict() must letterbox like predict_batch ---

def test_predict_with_explicit_buckets_matches_predict_batch_single():
    """A 1080x1920 portrait through buckets=[(1024,1024)] must produce the
    SAME mask via predict() as via predict_batch([img])[0]. The earlier
    predict() squashed the input to bucket aspect → distorted output."""
    pred = BiRefNetPredictor(
        _Identity(), device="cpu", dtype=None, normalize=False,
        buckets=[(64, 64)],  # square bucket
    )
    # Tall portrait (much taller than wide). Use small dims to keep test fast.
    img = Image.new("RGB", (32, 96), color=(0, 0, 0))
    # Add a single bright stripe at y=20 — it should appear at y≈20 in the mask
    # if aspect is preserved (letterbox), or shifted/stretched if squashed.
    arr = np.asarray(img, dtype=np.uint8).copy()
    arr[20, :, :] = 255
    img = Image.fromarray(arr)

    mask_predict = pred.predict(img).numpy()
    mask_batch = pred.predict_batch([img])[0].numpy()
    # The two paths should now produce numerically very close masks
    # (the only difference is letterbox padding direction; both paths
    # apply the SAME letterbox now after PERF-5 fix).
    assert mask_predict.shape == (96, 32) == mask_batch.shape
    np.testing.assert_allclose(mask_predict, mask_batch, atol=2e-2)


# --- MISS-2 / PERF-1: deform_conv autocast must produce bf16 output ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_deform_conv_autocast_returns_bf16_under_bf16_autocast():
    """The earlier fp32-fallback gate misfired (x.dtype was fp32 even under
    autocast because autocast doesn't touch the input). The fixed version
    derives target dtype from the offset conv's output (which IS autocast-
    promoted) and casts the result back to bf16.
    """
    from models.modules.deform_conv import DeformableConv2d
    m = DeformableConv2d(3, 8, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 3, 32, 32, device="cuda")
    eager = m(x)  # fp32 path
    assert eager.dtype == torch.float32
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        amp = m(x)
    # Output dtype matches the autocast-effective dtype (bf16), not fp32.
    assert amp.dtype == torch.bfloat16, (
        f"deform_conv under bf16 autocast returned {amp.dtype}; "
        f"the autocast gate is broken again."
    )
    # Numerical agreement (bf16 has ~7-bit mantissa).
    torch.testing.assert_close(eager, amp.float(), atol=5e-2, rtol=5e-2)


# --- BUG-1: 4D+ ndarray gives a useful error, not a confusing PIL crash ---

def test_load_pil_rejects_4d_ndarray_with_useful_error():
    from birefnet_api.predictor import _load_pil
    arr = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"unsupported numpy ndim=4"):
        _load_pil(arr)


# --- BUG-3: Counter tiebreak is deterministic across input order ---

def test_pick_batch_bucket_deterministic_tiebreak():
    """Two-bucket tie should resolve to the same bucket regardless of the
    input ordering — preventing torch.compile cache thrash from request
    ordering noise."""
    pred = BiRefNetPredictor(
        _Identity(), device="cpu", dtype=None, normalize=False,
        buckets=[(64, 64), (96, 96)],
    )
    items_hw_a = [(64, 64), (96, 96)]  # one of each → tie
    items_hw_b = [(96, 96), (64, 64)]  # reversed
    bucket_a = pred._pick_batch_bucket(items_hw_a, max_edge=128)
    bucket_b = pred._pick_batch_bucket(items_hw_b, max_edge=128)
    assert bucket_a == bucket_b, "tiebreak depends on input order"


# --- BUG-9: OOM handler doesn't crash on secondary CUDA failures ---

def test_oom_handler_robust_to_empty_cache_failure(monkeypatch):
    """If empty_cache() itself raises (e.g. corrupted CUDA context), the
    OOM error message still surfaces with the original exception. Patch
    the inner model.forward so the OOM propagates THROUGH _forward's
    try/except wrapper (patching _forward itself would bypass it)."""
    class _OOMModel(nn.Module):
        def forward(self, x):
            raise torch.cuda.OutOfMemoryError("synthetic OOM for test")
    pred = BiRefNetPredictor(_OOMModel(), device="cpu", dtype=None, normalize=False)
    # Make empty_cache itself blow up to simulate driver corruption
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: (_ for _ in ()).throw(RuntimeError("driver corrupt")))
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        pred.predict(img)
    # Original message must be in the chain
    assert "synthetic OOM for test" in str(excinfo.value)
    assert "OOM during forward" in str(excinfo.value)


# --- BUG-11: _pick_batch_bucket caps long axis to max_edge ---

def test_pick_batch_bucket_caps_to_max_edge():
    """A batch of (12000x6000) + (6000x12000) should not produce a bucket
    bigger than max_edge x max_edge — the per-axis max would otherwise
    stack them into a square."""
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=2048)
    bh, bw = pred._pick_batch_bucket([(12000, 6000), (6000, 12000)], max_edge=2048)
    assert max(bh, bw) <= 2048, f"bucket {(bh, bw)} exceeds max_edge=2048"


# --- BUG-10: predict_batch crop math handles m_h != bh edge case ---

def test_predict_batch_handles_letterbox_without_crash():
    """A batch of differently-aspected inputs hits the letterbox crop math.
    The fixed code clamps slice indices and ensures end > start, so a
    future m_h != bh case can't produce an empty slice."""
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=64)
    images = [
        Image.new("RGB", (32, 96), color=(0, 100, 200)),  # tall
        Image.new("RGB", (96, 32), color=(200, 100, 0)),  # wide
    ]
    masks = pred.predict_batch(images)
    assert len(masks) == 2
    assert masks[0].shape == (96, 32)
    assert masks[1].shape == (32, 96)
    for m in masks:
        assert torch.all(torch.isfinite(m))


# --- MISS-3: concurrent predict() calls thread-safe ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="threading test only meaningful on CUDA")
def test_concurrent_predict_thread_safe():
    """8 threads hitting predict() concurrently must all return matching
    masks (same input → same output) without crashing on cache races."""
    import concurrent.futures
    pred = BiRefNetPredictor(_Identity(), device="cpu", dtype=None, normalize=False, max_edge=64)
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    def call(): return pred.predict(img)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = [f.result() for f in [ex.submit(call) for _ in range(32)]]
    for r in results:
        assert r.shape == (64, 64)
        torch.testing.assert_close(r, results[0])


# --- MISS-7: OOM error message contains diagnostic fields ---

def test_oom_message_format():
    """Production debugging relies on the OOM message naming the input shape,
    dtype, max_edge, and the free/total memory. Patch the inner model.forward
    so the wrapper actually fires."""
    class _OOMModel(nn.Module):
        def forward(self, x):
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
    pred = BiRefNetPredictor(_OOMModel(), device="cpu", dtype=None, normalize=False, max_edge=64)
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        pred.predict(img)
    msg = str(excinfo.value)
    # Required diagnostic fields:
    assert "OOM during forward at input" in msg
    assert "dtype=" in msg
    assert "max_edge" in msg
    assert "synthetic OOM" in msg
