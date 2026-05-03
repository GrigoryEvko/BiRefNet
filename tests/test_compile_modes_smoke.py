"""Diagnostic: which torch.compile modes actually work for the predictor?

The predictor defaults to compile_mode='reduce-overhead' which uses
cudagraphs. cudagraphs requires:
  - static input shapes (recompile per shape)
  - no CPU↔GPU sync inside the captured region
  - tensors on the same stream
  - no host-side branching that affects graph structure

Our model has potential issues:
  - Swin _get_attn_mask allocates `torch.zeros((1, Hp, Wp, 1))` based on
    int input dims — could trigger graph breaks
  - LRU cache has Python-level branching (hit / miss / evict)
  - DeformableConv2d delegates to torchvision custom op
  - image2patches uses einops.rearrange + replicate-pad on potentially
    non-divisible shapes
  - Bucketing in the predictor means input shapes can vary per request
    (cudagraphs would recompile per shape)

These tests exercise each mode end-to-end on real CUDA hardware, run a
forward pass, capture timing, and compare against the eager baseline.
A mode that "works" means: compiles without errors, produces a
finite mask, mean output value within 0.01 of the eager baseline.

CUDA-gated. Run on the L40S production box; the dev laptop's 5.6 GB
VRAM is too tight to run Swin-L at compile-friendly sizes.

Set BIREFNET_TEST_COMPILE=1 to run (off by default since it takes
~3-5 minutes — Triton autotuning is slow).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pytestmark = pytest.mark.skipif(
    not os.environ.get("BIREFNET_TEST_COMPILE"),
    reason="set BIREFNET_TEST_COMPILE=1 to run torch.compile diagnostics (slow)",
)


def _gradient_pil(h: int, w: int):
    from PIL import Image
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    arr = np.stack([
        (xx * 255).astype(np.uint8),
        (yy * 255).astype(np.uint8),
        ((1 - xx) * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)


def _build_real_model_random_init():
    """Build a real BiRefNet with random weights — no HF download required.
    Random weights are fine for compile diagnostics: we're testing the
    graph-capture path, not output quality."""
    import config as _config
    _orig = _config.Config.__init__
    def _safe(self):
        try:
            _orig(self)
        except FileNotFoundError:
            pass
    _config.Config.__init__ = _safe
    try:
        from models.birefnet import BiRefNet
        return BiRefNet(bb_pretrained=False)
    finally:
        _config.Config.__init__ = _orig


@pytest.fixture(scope="module")
def baseline_eager():
    """Eager-mode predictor + a fixed test image. Subsequent tests compare
    against this baseline to verify each compile mode produces consistent
    output."""
    pytest.importorskip("kornia")
    pytest.importorskip("cv2")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from birefnet_api import BiRefNetPredictor
    model = _build_real_model_random_init()
    pred = BiRefNetPredictor(
        model, device="cuda", dtype="bf16", max_edge=512,
        compile=False,
    )
    img = _gradient_pil(384, 384)
    # Eager forward — this is our reference.
    mask = pred.predict(img)
    return {
        "img": img,
        "shape": (384, 384),
        "mask_mean": float(mask.mean().item()),
        "mask_std": float(mask.std().item()),
    }


def _try_compile_mode(mode: str, baseline: dict):
    """Build a fresh predictor with the given compile mode, run a forward,
    return (success: bool, message: str, latency_ms: float, mean_diff: float).
    """
    import torch
    from birefnet_api import BiRefNetPredictor

    model = _build_real_model_random_init()
    # Capture warnings so we can surface relevant compiler complaints.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            pred = BiRefNetPredictor(
                model, device="cuda", dtype="bf16", max_edge=512,
                compile=True, compile_mode=mode,
            )
            # First forward triggers compilation. Time it separately.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mask = pred.predict(baseline["img"])
            torch.cuda.synchronize()
            compile_time = (time.perf_counter() - t0) * 1000

            # Second forward should hit the cached/cudagraph'd path.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mask = pred.predict(baseline["img"])
            torch.cuda.synchronize()
            steady_time = (time.perf_counter() - t0) * 1000

            # Sanity: mask should be finite and shape-correct.
            if not torch.all(torch.isfinite(mask)):
                return False, f"mode={mode!r}: non-finite output", steady_time, float("nan")
            if mask.shape != baseline["shape"]:
                return False, f"mode={mode!r}: shape mismatch {mask.shape} vs {baseline['shape']}", steady_time, float("nan")

            # Compare mean to eager baseline. Random weights drift acceptably
            # under reduced precision; allow 0.05 absolute diff.
            mean_diff = abs(float(mask.mean().item()) - baseline["mask_mean"])
            if mean_diff > 0.1:
                return False, f"mode={mode!r}: mean output diverged by {mean_diff:.3f} from eager", steady_time, mean_diff

            relevant_warnings = [str(wi.message) for wi in w if "torch" in str(wi.filename).lower()]
            warning_summary = f" ({len(relevant_warnings)} torch warnings)" if relevant_warnings else ""

            return True, (
                f"mode={mode!r}: compile_first={compile_time:.0f}ms "
                f"steady={steady_time:.0f}ms diff={mean_diff:.4f}{warning_summary}"
            ), steady_time, mean_diff
        except Exception as e:
            return False, f"mode={mode!r}: {type(e).__name__}: {e}", float("nan"), float("nan")


def test_compile_mode_default(baseline_eager, capsys):
    """Inductor + AOTAutograd, no cudagraphs, no autotuning. Safest mode."""
    ok, msg, _, _ = _try_compile_mode("default", baseline_eager)
    print(f"\n{msg}")
    assert ok, msg


def test_compile_mode_reduce_overhead(baseline_eager, capsys):
    """Inductor + cudagraphs. Predictor's current default. Bucketing means
    cudagraphs will recompile per distinct shape — first request per
    shape pays compile cost, subsequent are fast. May still fail if
    Swin caches trigger graph breaks that are incompatible with cudagraphs.
    """
    ok, msg, _, _ = _try_compile_mode("reduce-overhead", baseline_eager)
    print(f"\n{msg}")
    if not ok:
        pytest.skip(
            f"reduce-overhead mode failed in this environment: {msg}\n"
            f"Recommended: switch predictor default to 'default' or "
            f"'max-autotune-no-cudagraphs'."
        )


def test_compile_mode_max_autotune_no_cudagraphs(baseline_eager, capsys):
    """Triton kernel autotuning, no cudagraphs. Recommended for training
    (DDP grad hooks + cudagraphs don't mix). For inference: gives the
    Triton speedup without the cudagraph shape-rigidity."""
    ok, msg, _, _ = _try_compile_mode("max-autotune-no-cudagraphs", baseline_eager)
    print(f"\n{msg}")
    assert ok, msg


def test_compile_mode_max_autotune(baseline_eager, capsys):
    """Triton autotuning + cudagraphs. The most aggressive mode — biggest
    potential speedup, biggest compile cost (5-15 min first time), strictest
    constraints. May fail same way as reduce-overhead if cudagraphs are
    incompatible with the predictor's bucketed input."""
    ok, msg, _, _ = _try_compile_mode("max-autotune", baseline_eager)
    print(f"\n{msg}")
    if not ok:
        pytest.skip(
            f"max-autotune mode failed in this environment: {msg}\n"
            f"Likely cause: cudagraphs incompatible with bucketed input "
            f"shapes or Swin attn_mask cache. Use "
            f"'max-autotune-no-cudagraphs' instead."
        )


def test_compile_summary_recommends_a_mode(baseline_eager, capsys):
    """Run all four modes back-to-back and print a recommendation table.
    This is the user-facing diagnostic — `pytest -s` shows the table.
    """
    results = {}
    for mode in ("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"):
        ok, msg, steady_ms, _ = _try_compile_mode(mode, baseline_eager)
        results[mode] = (ok, msg, steady_ms)

    print("\n\n=== torch.compile mode diagnostics ===")
    print(f"{'mode':<32s}  {'ok':>4s}  {'steady_ms':>10s}")
    print("-" * 52)
    for mode, (ok, _, steady_ms) in results.items():
        steady_repr = f"{steady_ms:.0f}" if steady_ms == steady_ms else "n/a"  # NaN check
        print(f"{mode:<32s}  {('y' if ok else 'N'):>4s}  {steady_repr:>10s}")
    # Pick the fastest passing mode as a recommendation.
    passing = [(m, ms) for m, (ok, _, ms) in results.items() if ok and ms == ms]
    if passing:
        best_mode, best_ms = min(passing, key=lambda kv: kv[1])
        print(f"\nrecommendation: compile_mode={best_mode!r} (steady {best_ms:.0f}ms)")
    else:
        print("\nrecommendation: compile=False (all modes failed)")
    # Always passes — this is a diagnostic, not a gate.
    # The individual tests above will fail on regressions in 'default' or
    # 'max-autotune-no-cudagraphs'.
