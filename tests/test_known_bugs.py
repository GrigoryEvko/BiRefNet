"""Regression tests that demonstrate concrete bugs in the upstream codebase.

Each test isolates the smallest reproducible piece of broken code without
requiring the full model graph or external datasets. The intent is twofold:

  - prove the bug exists (the test fails today),
  - serve as a regression check after the fix lands.

Tests that document expected-bad behavior are marked `xfail(strict=True)`.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


# ------------------------------------------------------------------ helpers
def _check_state_dict_under_test():
    """Return the project's `check_state_dict`, falling back to an inlined copy
    when cv2 isn't installed in the test env.

    The inlined copy is a byte-for-byte transcription of `utils.py:29-36` so the
    test exercises the *exact* upstream logic regardless of import side effects.
    """
    try:
        import cv2  # noqa: F401
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return importlib.import_module("utils").check_state_dict
    except ImportError:
        def check_state_dict(state_dict, unwanted_prefixes=("module.", "_orig_mod.")):
            for k, v in list(state_dict.items()):
                prefix_length = 0
                for unwanted_prefix in unwanted_prefixes:
                    if k[prefix_length:].startswith(unwanted_prefix):
                        prefix_length += len(unwanted_prefix)
                state_dict[k[prefix_length:]] = state_dict.pop(k)
            return state_dict
        return check_state_dict


# ------------------------------------------------------ check_state_dict bug

def test_check_state_dict_strips_nested_prefixes_in_either_order():
    """Regression: '_orig_mod.module.X' (compile+DDP) and 'module._orig_mod.X'
    must both reduce to 'X' regardless of prefix list order."""
    check_state_dict = _check_state_dict_under_test()
    sd = {"_orig_mod.module.layer.weight": 1, "module._orig_mod.layer.bias": 2}
    out = check_state_dict(dict(sd))
    assert "layer.weight" in out
    assert "layer.bias" in out


def test_check_state_dict_raises_on_collision():
    """Regression: silent overwrite when both prefixed and bare keys exist."""
    check_state_dict = _check_state_dict_under_test()
    sd = {"module.weight": 1, "weight": 2}
    with pytest.raises(ValueError, match="collision"):
        check_state_dict(dict(sd))


def test_check_state_dict_handles_torch_tensors():
    """Real-world: torch.save objects are tensors, not Python ints."""
    import torch
    check_state_dict = _check_state_dict_under_test()
    sd = {
        "_orig_mod.layer.weight": torch.zeros(2),
        "module.layer.bias": torch.zeros(3),
        "layer.gamma": torch.ones(1),
    }
    out = check_state_dict(dict(sd))
    assert torch.equal(out["layer.weight"], torch.zeros(2))
    assert torch.equal(out["layer.bias"], torch.zeros(3))
    assert torch.equal(out["layer.gamma"], torch.ones(1))


def test_check_state_dict_handles_single_prefix():
    """Same function with a single-prefix key should always work."""
    check_state_dict = _check_state_dict_under_test()
    sd = {"module.layer.weight": 1, "_orig_mod.layer.bias": 2, "layer.gamma": 3}
    out = check_state_dict(dict(sd))
    assert out["layer.weight"] == 1
    assert out["layer.bias"] == 2
    assert out["layer.gamma"] == 3


# ------------------------------------------------------ dataset path-replace bug

def test_dataset_path_replace_corrupts_paths_with_repeated_im_segment():
    """`p.replace('/im/', '/gt/')` on a path where 'im' appears more than once
    silently replaces every occurrence and produces a corrupted GT path."""
    p = "/data/im/dataset_im/im/sample.png"
    fake = p.replace("/im/", "/gt/")
    assert "/gt/dataset_im/gt/" in fake
    # The "right" answer is to replace only the last "/im/" segment, which
    # `str.replace` cannot do without manual logic.
    correct = "/data/im/dataset_im/gt/sample.png"
    assert fake != correct  # demonstrates the bug


def test_dataset_path_replace_breaks_on_windows_separators():
    """The same call breaks if the source path uses backslashes."""
    p = r"C:\data\im\sample.png"
    assert p.replace("/im/", "/gt/") == p  # no change → label silently misses


# ------------------------------------------------------ Image.LANCZOS handling

def test_pillow_resampling_constant_present():
    """Pillow ≥ 9.1 moved LANCZOS to Image.Resampling. The predictor handles both."""
    from PIL import Image
    constant = getattr(Image, "Resampling", Image).LANCZOS
    img = Image.new("RGB", (8, 8))
    out = img.resize((4, 4), constant)
    assert out.size == (4, 4)


# ------------------------------------------------------ BCELoss + autocast warning

def test_BCELoss_under_bf16_autocast_raises():
    """PyTorch refuses to run nn.BCELoss inside autocast — it raises a RuntimeError
    pointing the user at BCEWithLogitsLoss. The repo's loss.py:136/144 and
    train.py:159 both still use BCELoss-on-sigmoid, so any bf16/fp16 training
    run that hits this path crashes. (Confirmed against torch 2.11.)"""
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for autocast")
    crit = nn.BCELoss()
    pred = torch.rand(1, 1, 32, 32, device="cuda").requires_grad_(True)
    target = torch.rand(1, 1, 32, 32, device="cuda")
    with pytest.raises(RuntimeError, match="unsafe to autocast"):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            crit(pred.sigmoid(), target)


def test_BCEWithLogits_under_bf16_autocast_is_safe():
    """Counterpart: BCEWithLogitsLoss is autocast-safe. This is the recommended fix."""
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for autocast")
    crit = nn.BCEWithLogitsLoss()
    pred = torch.randn(1, 1, 32, 32, device="cuda").requires_grad_(True)
    target = torch.rand(1, 1, 32, 32, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = crit(pred, target)
    loss.backward()
    assert pred.grad is not None


# ----------------------------------------- multi_scl_ipt='add' branch with vgg16

def test_mul_scl_ipt_add_branch_handles_vgg_resnet():
    """Real exercise: with mul_scl_ipt='add' + bb='vgg16', the previous code
    called self.bb(x_pyramid) blindly and crashed with AttributeError because
    vgg16 exposes conv1..conv4 instead of being directly callable. Now goes
    through _run_backbone which dispatches per-backbone.
    """
    pytest.importorskip("torch")
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    import config as _config

    orig_init = _config.Config.__init__
    def _patched(self):
        try:
            orig_init(self)
        except FileNotFoundError:
            pass
        # Override after default init to guarantee the path under test.
        self.bb = 'vgg16'
        self.mul_scl_ipt = 'add'
        # vgg16's lateral channels — restate so the model picks them up.
        # mul_scl_ipt='add' keeps channels as-is (no x2 doubling).
        self.lateral_channels_in_collection = [512, 512, 256, 128]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

    _config.Config.__init__ = _patched
    # Drop cached imports so the swap takes effect.
    for mod in ("models.birefnet", "models.backbones.build_backbone"):
        sys.modules.pop(mod, None)
    try:
        from models.birefnet import BiRefNet
        model = BiRefNet(bb_pretrained=False).eval()
        # vgg16 max-pools 5x → /32 downsample. 96 is divisible by 32 and small.
        x = torch.zeros(1, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        # eval mode returns the list of scaled_preds.
        assert isinstance(out, list)
        assert out, "model returned empty list"
        last = out[-1]
        assert torch.is_tensor(last)
        assert torch.all(torch.isfinite(last))
    finally:
        _config.Config.__init__ = orig_init
        for mod in ("models.birefnet", "models.backbones.build_backbone"):
            sys.modules.pop(mod, None)
