"""Regression tests for dynamic_size sampling:

  - tuple(sorted(dynamic_size)) must NOT be applied (it swapped W/H ranges)

Each rank samples independently — DDP all-reduces gradients (parameter shape),
not activations, so different per-rank input sizes don't trigger an all-reduce
shape mismatch.
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
    # Try a case where sort would actually flip:
    dynamic_size = ((150, 300), (50, 200))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)
    random.seed(0)
    w, h = ds._sample_dynamic_size()
    # If the sort was still in place, w would be sampled from (50, 200)
    # because lex-sort puts (50, 200) first. We want w in (150, 300).
    assert 128 <= w <= 320, w
    assert 32 <= h <= 224, h


def test_independent_sampling_returns_div_32_multiples(monkeypatch):
    """Sampled (W, H) are always multiples of 32 within the configured range."""
    dynamic_size = ((128, 512), (128, 512))
    ds = _import_dataset_with_stub_config(monkeypatch, dynamic_size)
    random.seed(123)
    for _ in range(50):
        w, h = ds._sample_dynamic_size()
        assert w % 32 == 0
        assert h % 32 == 0
        assert 128 <= w <= 512
        assert 128 <= h <= 512
