"""Regression tests for dynamic_size sampling:

  - tuple(sorted(dynamic_size)) must NOT be applied (it swapped W/H ranges)
  - the sampled batch shape must be the same on every DDP rank (else NCCL
    all-reduce shape mismatch)

These tests don't spin up real NCCL — they patch torch.distributed to
simulate two ranks and verify the broadcast path is exercised.
"""
from __future__ import annotations

import importlib
import os
import random
import sys


def _import_dataset_with_stub_config(monkeypatch, dynamic_size):
    """Reload dataset.py with a stubbed Config. Returns the module."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as _config

    class _Stub:
        def __init__(self):
            self.task = "DIS5K"
            self.testsets = ""
            self.training_set = ""
            self.size = (1024, 1024)
            self.dynamic_size = dynamic_size
            self.background_color_synthesis = False
            self.preproc_methods = []
            self.auxiliary_classification = False
            self.data_root_dir = "/tmp/nope"
            self.load_all = False
            self.device = "cpu"

    monkeypatch.setattr(_config, "Config", lambda: _Stub())
    if "dataset" in sys.modules:
        del sys.modules["dataset"]
    return importlib.import_module("dataset")


def test_dropped_sorted_keeps_dimension_order(monkeypatch):
    """Without sort: w_range=(50,300), h_range=(100,200) → sample uses
    those *as given*, not lexicographically sorted."""
    dynamic_size = ((50, 300), (100, 200))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)
    random.seed(0)
    w, h = ds._sample_dynamic_size()
    # Outputs are snapped to /32 multiples within the *original* ranges.
    assert 32 <= w <= 320, w
    assert 96 <= h <= 224, h
    # Critical: w must come from the W range (50..300), h from the H range.
    # Pre-fix code would sort the tuple-of-tuples lex and swap them when
    # 50 < 100 → ((50,300),(100,200)) sorted is ((50,300),(100,200)) — same.
    # Try a case where sort would actually flip:
    dynamic_size = ((150, 300), (50, 200))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)
    random.seed(0)
    w, h = ds._sample_dynamic_size()
    # If the sort was still in place, w would be sampled from (50, 200)
    # because lex-sort puts (50, 200) first. We want w in (150, 300).
    assert 128 <= w <= 320, w
    assert 32 <= h <= 224, h


def test_ddp_broadcast_path_invoked(monkeypatch):
    """Simulate dist.is_initialized() == True with world_size==2 rank==0;
    verify dist.broadcast is called and returns the same size both ranks."""
    dynamic_size = ((128, 256), (128, 256))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)

    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    broadcast_calls = []
    def fake_broadcast(tensor, src=0):
        broadcast_calls.append((tensor.clone(), src))
    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    random.seed(42)
    w, h = ds._sample_dynamic_size()
    assert len(broadcast_calls) == 1
    bcast_tensor, src = broadcast_calls[0]
    assert src == 0
    assert int(bcast_tensor[0].item()) == w
    assert int(bcast_tensor[1].item()) == h
    assert w % 32 == 0 and h % 32 == 0


def test_non_rank_0_receives_size_from_broadcast(monkeypatch):
    """Rank 1 starts with (0, 0) and receives the broadcast value."""
    dynamic_size = ((128, 256), (128, 256))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)

    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    def fake_broadcast(tensor, src=0):
        # Pretend rank 0 sent (160, 224)
        tensor[0] = 160
        tensor[1] = 224
    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    w, h = ds._sample_dynamic_size()
    assert (w, h) == (160, 224)
