"""Regression test: freeze_bb must also freeze BN running stats.

requires_grad=False stops gradient updates but BatchNorm running_mean and
running_var are buffers — they get updated on every training-mode forward.
With freeze_bb=True on a BN backbone (vgg16/resnet50), these stats drifted
from the pretrained values during training, silently corrupting the model.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")
pytest.importorskip("cv2")


def _build_model_with_freeze_bb(bb_name='resnet50'):
    """Build a BiRefNet with freeze_bb=True. resnet50 is used because it has
    BatchNorm layers — vgg16 (no _bn) is plain Conv+ReLU and exercises a
    different code path."""
    import config as _config
    orig_init = _config.Config.__init__

    # resnet50 channels.
    bb_channels = {
        'resnet50': [2048, 1024, 512, 256],
        'vgg16bn': [512, 512, 256, 128],
    }[bb_name]

    def _patched(self):
        try:
            orig_init(self)
        except FileNotFoundError:
            pass
        self.bb = bb_name
        self.freeze_bb = True
        # Disable mul_scl_ipt so channel arithmetic stays simple.
        self.mul_scl_ipt = ''
        self.lateral_channels_in_collection = bb_channels
        self.cxt = (self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:]
                    if self.cxt_num else [])

    _config.Config.__init__ = _patched
    for mod in ("models.birefnet", "models.backbones.build_backbone"):
        sys.modules.pop(mod, None)
    try:
        from models.birefnet import BiRefNet
        return BiRefNet(bb_pretrained=False)
    finally:
        _config.Config.__init__ = orig_init
        for mod in ("models.birefnet", "models.backbones.build_backbone"):
            sys.modules.pop(mod, None)


def test_freeze_bb_keeps_backbone_bn_in_eval_during_train():
    """When parent .train() is called, backbone BN must stay in eval mode."""
    import torch.nn as nn
    model = _build_model_with_freeze_bb('resnet50')
    model.train()  # parent in train mode
    bb_train = [m.training for m in model.bb.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bb_train, "resnet50 backbone has no BN layers — wrong fixture"
    assert not any(bb_train), "frozen backbone BN is in train mode — running stats will drift"
    assert model.training is True


def test_freeze_bb_running_stats_unchanged_after_train_forward():
    """A training-mode forward must NOT mutate frozen-backbone BN running stats."""
    import torch.nn as nn
    model = _build_model_with_freeze_bb('resnet50')
    model.train()
    snapshots = []
    for m in model.bb.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            snapshots.append((m, m.running_mean.clone(), m.running_var.clone()))
    assert snapshots, "no BN in resnet50 backbone — wrong fixture"

    # Random non-zero input — zeros input doesn't update running stats.
    torch.manual_seed(0)
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        model(x)

    for m, prev_mean, prev_var in snapshots:
        assert torch.equal(m.running_mean, prev_mean), "BN running_mean drifted under freeze_bb"
        assert torch.equal(m.running_var, prev_var), "BN running_var drifted under freeze_bb"
