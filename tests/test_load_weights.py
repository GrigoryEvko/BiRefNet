"""Regression test: load_weights drops mismatched-shape keys with a log,
instead of silently substituting random init weights from model_dict."""
from __future__ import annotations

import os
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def filter_state_dict():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.backbones.build_backbone import _filter_state_dict
    return _filter_state_dict


def test_drops_mismatched_size_and_reports(filter_state_dict):
    save_model = {
        "conv.weight": torch.zeros(64, 3, 3, 3),  # ckpt has 64 out channels
        "bn.weight": torch.zeros(64),
        "extra_unused": torch.zeros(8),
    }
    model_dict = {
        "conv.weight": torch.zeros(96, 3, 3, 3),  # model expects 96
        "bn.weight": torch.zeros(96),
        "fc.weight": torch.zeros(10, 96),
    }
    state_dict, dropped = filter_state_dict(save_model, model_dict)
    # Both ckpt keys are present in the model but with mismatched shape:
    assert state_dict == {}
    dropped_names = [d[0] for d in dropped]
    assert "conv.weight" in dropped_names
    assert "bn.weight" in dropped_names
    # Unused extra keys not present in the model are silently skipped (no log spam):
    assert "extra_unused" not in dropped_names


def test_keeps_matching_keys(filter_state_dict):
    save_model = {"conv.weight": torch.ones(8, 3, 3, 3), "ignore_me": torch.zeros(1)}
    model_dict = {"conv.weight": torch.zeros(8, 3, 3, 3), "fc.weight": torch.zeros(10, 8)}
    state_dict, dropped = filter_state_dict(save_model, model_dict)
    assert "conv.weight" in state_dict
    assert torch.equal(state_dict["conv.weight"], torch.ones(8, 3, 3, 3))
    assert dropped == []


def test_no_silent_random_substitution(filter_state_dict):
    """The previous bug: when shapes don't match, the dict comprehension
    used `model_dict[k]` (i.e. random init) — so the model loaded with a
    pretrained mix of partly-fresh weights, with no log line."""
    save_model = {"conv.weight": torch.zeros(8, 3, 3, 3)}
    model_dict = {"conv.weight": torch.full((16, 3, 3, 3), 99.0)}
    state_dict, _ = filter_state_dict(save_model, model_dict)
    # Critically: no entry in state_dict that points at model_dict's value.
    assert "conv.weight" not in state_dict
