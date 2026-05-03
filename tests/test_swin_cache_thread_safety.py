"""Concurrency tests for Swin caches.

The LRU update + eviction sequence on _attn_mask_cache and the
double-checked insert on _rpb_cache must be safe under threaded forward
calls — FastAPI's run_in_threadpool, uvicorn --workers, or any custom
thread-pool serving setup all hit this code from multiple threads on
the same predictor instance.
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


def _make_layer(depth=2, num_heads=2, dim=8, window_size=7):
    from models.backbones.swin_v1 import BasicLayer
    return BasicLayer(dim=dim, depth=depth, num_heads=num_heads,
                      window_size=window_size).eval()


def _make_attn(window=7, num_heads=2, dim=8):
    from models.backbones.swin_v1 import WindowAttention
    return WindowAttention(dim=dim, window_size=(window, window), num_heads=num_heads)


def test_attn_mask_cache_concurrent_same_shape():
    """Many threads requesting the same shape: all must get a valid mask
    and the cache stays consistent (no KeyError, no None)."""
    layer = _make_layer()
    cpu = torch.device("cpu")
    barrier = threading.Barrier(16)

    def _worker(_):
        barrier.wait()  # all threads pile into the call simultaneously
        return layer._get_attn_mask(28, 28, torch.float32, cpu)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(_worker, range(16)))
    assert all(r is not None for r in results)
    # All threads see the same final cached tensor (after the dust settles).
    assert all(torch.equal(r, results[0]) for r in results)
    # Cache holds exactly one entry.
    assert len(layer._attn_mask_cache) == 1


def test_attn_mask_cache_concurrent_different_shapes():
    """Threads requesting many different shapes hammer the LRU eviction
    path — must not raise KeyError on move_to_end / popitem races."""
    layer = _make_layer()
    cpu = torch.device("cpu")
    shapes = [(28, 28), (35, 35), (42, 42), (49, 49), (56, 56), (63, 63), (70, 70), (77, 77)]
    # Multiply shapes so each gets hit many times.
    work = shapes * 8
    barrier = threading.Barrier(len(work))

    errors = []
    def _worker(hw):
        try:
            barrier.wait()
            return layer._get_attn_mask(hw[0], hw[1], torch.float32, cpu)
        except Exception as e:
            errors.append(e)
            return None

    with ThreadPoolExecutor(max_workers=len(work)) as ex:
        results = list(ex.map(_worker, work))
    assert not errors, f"races raised: {errors[:3]}"
    assert all(r is not None for r in results)
    # Cache cap is 4; never exceeds it.
    assert len(layer._attn_mask_cache) <= 4


def test_rpb_cache_concurrent_same_dtype():
    """Many threads requesting the same dtype/device key — only one entry
    inserted, all callers see the same tensor."""
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    barrier = threading.Barrier(16)

    def _worker(_):
        barrier.wait()
        return attn._relative_position_bias(q)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(_worker, range(16)))
    assert all(r is not None for r in results)
    # All threads must see the SAME tensor object (the winner of the race).
    assert all(r is results[0] for r in results), "concurrent inserts produced multiple cached tensors"
    assert len(attn._rpb_cache) == 1


def test_rpb_cache_train_clear_under_concurrent_load():
    """A .train() call clears the cache; concurrent forward()s after must
    rebuild rather than crash on the empty dict."""
    attn = _make_attn().eval()
    q = torch.zeros(1, 2, 49, 4)
    attn._relative_position_bias(q)  # populate

    errors = []
    def _worker(_):
        try:
            return attn._relative_position_bias(q)
        except Exception as e:
            errors.append(e)
            return None

    # Concurrent forwards while train() / eval() flips
    def _flipper():
        for _ in range(50):
            attn.train()
            attn.eval()

    flip_thread = threading.Thread(target=_flipper)
    flip_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_worker, range(50)))
    finally:
        flip_thread.join()
    assert not errors, f"flip vs forward race: {errors[:3]}"
    assert all(r is not None for r in results)
