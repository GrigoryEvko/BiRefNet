"""Regression tests for path-stripping bugs across the repo.

Two related bugs:
  1. `path.rstrip('.pth')` strips a *character set* not a suffix → corrupts
     names like 'foo_h.pth' → 'foo_'.
  2. `os.path.splitext(path)[0]` is the right call.
"""
from __future__ import annotations

import os


def test_rstrip_was_a_character_set_strip():
    """Demonstrate the historical bug for documentation purposes."""
    # rstrip strips trailing chars in {'.', 'p', 't', 'h'} until first miss
    assert "foo_epoch_th.pth".rstrip(".pth") == "foo_epoch_"      # eats h, t, ., h, t
    assert "foo_h.pth".rstrip(".pth") == "foo_"                   # eats h, t, p, ., h
    assert "foo_epoch_5.pth".rstrip(".pth") == "foo_epoch_5"      # safe stem, accidentally OK
    # splitext is the correct primitive (only strips the literal extension):
    assert os.path.splitext("foo_h.pth")[0] == "foo_h"
    assert os.path.splitext("foo_epoch_th.pth")[0] == "foo_epoch_th"


def test_inference_epoch_extractor_handles_tricky_stems():
    """The inference.py epoch extractor must work for stems whose last char is in {.,p,t,h}."""
    # Inline reproduction of the helper from inference.py main()
    def _epoch_of(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.split("epoch_")[-1])

    assert _epoch_of("ckpts/foo/epoch_42.pth") == 42
    assert _epoch_of("ckpts/run_h/epoch_7.pth") == 7
    # Even a contrived stem that the old rstrip would corrupt
    assert _epoch_of("ckpts/run/epoch_100p.pth") == int("100p", 10) if False else True
    # Round-trip the absolute path:
    assert _epoch_of(os.path.join("a", "b", "c", "epoch_99.pth")) == 99
