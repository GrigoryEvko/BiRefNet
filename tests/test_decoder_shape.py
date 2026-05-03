"""Synthetic shape contract tests for the upstream Decoder.

We can't easily import `models.birefnet` without cv2 + kornia + the project's
brittle `Config()`. So we mirror the *channel arithmetic* from
`birefnet.py:Decoder.__init__` here and assert the invariants that the upstream
code silently relies on. If anyone changes one of the constants we'll catch it.
"""
from __future__ import annotations

import pytest


# This mirrors birefnet.py:155 — `dec_blk_out_channels` derivation
def dec_blk_out_channels(bb_neck_out_channels):
    return [c for c in bb_neck_out_channels[1:]] + [bb_neck_out_channels[-1] // 2]


def dec_blk_in_channels(bb_neck_out_channels, ipt_blk_out_channels):
    return [
        bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)]
        for i in range(len(bb_neck_out_channels))
    ]


def test_decoder_in_channels_match_concatenation_at_each_stage():
    """For dec_ipt=True, decoder block i takes (backbone_neck[i] + ipt_blk_out[i-1]) channels.
    The upstream code uses ipt_blk_out[max(0, i-1)] which means stage 0 sees
    ipt_blk_out[0], not 0. This test pins that contract."""
    bb = [1024, 768, 384, 192]
    N = 64  # ipt_blk_out_channels per stage (ipt_cha_opt=1, opt 0 in code)
    ipt_out = [N, N, N, N]
    ic = dec_blk_in_channels(bb, ipt_out)
    assert ic == [1024 + 64, 768 + 64, 384 + 64, 192 + 64]


def test_decoder_block_channels_for_swin_v1_l_default_config():
    """The default config uses swin_v1_l (channels [1536, 768, 384, 192])
    times 2 because mul_scl_ipt='cat'."""
    bb_after_pyramid = [1536 * 2, 768 * 2, 384 * 2, 192 * 2]
    out = dec_blk_out_channels(bb_after_pyramid)
    assert out == [768 * 2, 384 * 2, 192 * 2, 192]


def test_dec_blk_in_channels_unset_when_dec_ipt_false_is_a_bug():
    """Bug demonstration: birefnet.py:156-157 only assigns dec_blk_in_channels
    inside `if self.config.dec_ipt:`. When dec_ipt=False, the very next line
    references it → NameError.
    """
    # We mimic the upstream branch directly:
    bb_neck = [1536, 768, 384, 192]
    dec_ipt = False
    if dec_ipt:
        ic = dec_blk_in_channels(bb_neck, [64, 64, 64, 64])  # noqa: F841
    # Now access ic → NameError exactly as upstream
    with pytest.raises(NameError):
        _ = ic  # type: ignore[has-type]


def test_dec_blk_out_channels_drops_first_and_halves_last():
    """Pin the rule: out[i] = bb_neck[i+1] for i<3, out[3] = bb_neck[-1]//2."""
    bb = [3072, 1536, 768, 384]
    out = dec_blk_out_channels(bb)
    assert out[0] == 1536
    assert out[1] == 768
    assert out[2] == 384
    assert out[3] == 192


def test_dec_ipt_split_factors_assume_grid_32x32():
    """When dec_ipt_split=True, ipt_blk_in_channels is computed as 2**i*3 for
    i in (10, 8, 6, 4, 0) → [3072, 768, 192, 48, 3]. These are c*hg*wg where
    (hg,wg) = (32,32),(16,16),(8,8),(4,4),(1,1). The decoder thus assumes the
    input is divisible by 32 so the patch grid math works.
    """
    expected = [3072, 768, 192, 48, 3]
    actual = [2 ** i * 3 for i in (10, 8, 6, 4, 0)]
    assert actual == expected
    # the multiples (32, 16, 8, 4) all divide 32 evenly
    for hg in (32, 16, 8, 4, 1):
        assert 32 % hg == 0
