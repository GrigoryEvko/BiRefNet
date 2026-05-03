"""Regression test: MBAMeasure.cal_ba must not divide by zero on flat GTs.

When the GT is entirely foreground or background, MORPH_GRADIENT yields an
empty boundary region. The previous code computed `num_pred_gd_pix / 0`
and injected NaN into the dataset-wide MBA average.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_cal_ba_returns_finite_on_all_foreground_gt():
    from evaluation.metrics import MBAMeasure
    m = MBAMeasure()
    gt = np.full((64, 64), 1, dtype=np.uint8)  # entirely foreground
    pred = np.full((64, 64), 1, dtype=np.uint8)
    ba = m.cal_ba(pred, gt)
    assert np.isfinite(ba), "cal_ba returned NaN on flat-foreground GT"


def test_cal_ba_returns_finite_on_all_background_gt():
    from evaluation.metrics import MBAMeasure
    m = MBAMeasure()
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    ba = m.cal_ba(pred, gt)
    assert np.isfinite(ba)


def test_cal_ba_normal_case_still_works():
    from evaluation.metrics import MBAMeasure
    m = MBAMeasure()
    gt = np.zeros((128, 128), dtype=np.uint8)
    gt[40:80, 40:80] = 1
    pred = gt.copy()
    ba = m.cal_ba(pred, gt)
    assert np.isfinite(ba)
    assert 0 <= ba <= 1
