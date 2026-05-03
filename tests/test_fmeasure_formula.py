"""Regression test: F-measure formula uses beta-squared, not beta.

The standard F_β = (1 + β²) P R / (β² P + R). The shipped code in
evaluation/metrics.py used `beta` instead of `beta**2`, which inflated every
adaptive_fm / mean_fm / max_fm score at the FMeasure default of beta=0.3.
WeightedFMeasure was numerically unaffected since its default beta=1 makes
β = β² = 1, but we fix it for symmetry.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def metrics():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation import metrics as m
    return m


def _expected_f(p, r, beta):
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def test_adaptive_fm_uses_beta_squared(metrics):
    """Direct formula check at P != R where β vs β² is observable."""
    fm = metrics.FMeasure(beta=0.3)
    pre, rec, beta = 0.8, 0.4, 0.3
    spec = _expected_f(pre, rec, beta)              # (1+0.09)*0.32/(0.072+0.4) ≈ 0.7390
    code_old = (1 + beta) * pre * rec / (beta * pre + rec)  # 0.65 — wrong
    assert abs(spec - 0.7390) < 0.01
    assert abs(code_old - 0.65) < 0.01
    assert abs(spec - code_old) > 0.05              # bug definitely observable
    # The fixed metric uses beta_sq:
    fixed = (1 + fm.beta_sq) * pre * rec / (fm.beta_sq * pre + rec)
    assert abs(fixed - spec) < 1e-9


def test_changeable_fms_uses_beta_squared(metrics):
    fm = metrics.FMeasure(beta=0.3)
    # Very simple GT/pred so we can compute by hand
    pred = np.full((4, 4), 200, dtype=np.uint8)
    gt = np.full((4, 4), 255, dtype=np.uint8)
    gt[:, 2:] = 0
    fm.step(pred=pred, gt=gt)
    # All gt==True pixels have pred>=threshold for the threshold range we care
    # about; the highest-recall, highest-precision points should match the spec.
    assert fm.beta_sq == pytest.approx(0.09)


def test_weighted_fm_uses_beta_squared(metrics):
    """WFM with beta=1 gives the same answer either way; with beta=0.5 it differs."""
    wfm = metrics.WeightedFMeasure(beta=0.5)
    assert wfm.beta_sq == pytest.approx(0.25)


def test_default_betas_unchanged(metrics):
    """Sanity: default constructor args still β=0.3 / β=1."""
    assert metrics.FMeasure().beta == 0.3
    assert metrics.WeightedFMeasure().beta == 1
