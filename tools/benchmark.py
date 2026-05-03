#!/usr/bin/env python
"""Predictor benchmark — measure latency + peak VRAM at production sizes.

    python tools/benchmark.py                       # random-init model, all sizes
    python tools/benchmark.py --pretrained          # download upstream from HF
    python tools/benchmark.py --sizes 1024,2048,4096 --warmup 5 --iters 30
    python tools/benchmark.py --device cpu --dtype fp32

Reports p50 / p90 / p99 latency and peak VRAM allocated for each input
size. Intended for capacity planning — figure out your kiosk SLA budget
on real hardware before serving real traffic.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from birefnet_api import BiRefNetPredictor


def _gradient_image(h: int, w: int) -> Image.Image:
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    arr = np.stack([
        (xx * 255).astype(np.uint8),
        (yy * 255).astype(np.uint8),
        ((1 - xx) * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)


def _build_predictor(args) -> BiRefNetPredictor:
    if args.pretrained:
        repo = args.repo or "ZhengPeng7/BiRefNet"
        print(f"Downloading {repo}...", flush=True)
        return BiRefNetPredictor.from_pretrained(
            repo, device=args.device, dtype=args.dtype, max_edge=args.max_edge,
        )
    # Random-init: useful for shape/timing without HF dependency.
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
        model = BiRefNet(bb_pretrained=False)
    finally:
        _config.Config.__init__ = _orig
    return BiRefNetPredictor(
        model, device=args.device, dtype=args.dtype, max_edge=args.max_edge,
    )


def _bench_one(pred: BiRefNetPredictor, h: int, w: int, warmup: int, iters: int):
    img = _gradient_image(h, w)
    is_cuda = pred.device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(pred.device)
    # Warmup — primes torch.compile, autotune, KV allocations.
    for _ in range(warmup):
        pred.predict(img)
    if is_cuda:
        torch.cuda.synchronize(pred.device)
    times = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize(pred.device)
        t0 = time.perf_counter()
        pred.predict(img)
        if is_cuda:
            torch.cuda.synchronize(pred.device)
        times.append(time.perf_counter() - t0)
    peak_mb = (
        torch.cuda.max_memory_allocated(pred.device) / 1e6 if is_cuda else float("nan")
    )
    return times, peak_mb


def _percentile(xs, p):
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true",
                    help="Download upstream weights from HF Hub.")
    ap.add_argument("--repo", default=None,
                    help="HF repo id when --pretrained (default ZhengPeng7/BiRefNet)")
    ap.add_argument("--device", default="auto", help="cuda / cpu / mps / auto")
    ap.add_argument("--dtype", default="bf16", help="bf16 / fp16 / fp32 / None")
    ap.add_argument("--max-edge", type=int, default=2048,
                    help="Internal bucket cap (model never sees more than this).")
    ap.add_argument("--sizes", default="1024,2048,4096,8192,12000",
                    help="Comma-separated WxH or single edge values.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if args.dtype.lower() in ("none", "fp32", "float32", "f32"):
        args.dtype = None

    pred = _build_predictor(args)
    print(repr(pred))
    print()
    print(f"{'shape':>15s}  {'p50':>8s}  {'p90':>8s}  {'p99':>8s}  {'mean':>8s}  {'peak_MB':>9s}")
    print("-" * 70)
    for tok in args.sizes.split(","):
        tok = tok.strip()
        if "x" in tok:
            h, w = (int(v) for v in tok.split("x"))
        else:
            h = w = int(tok)
        times, peak_mb = _bench_one(pred, h, w, args.warmup, args.iters)
        p50 = _percentile(times, 50) * 1000
        p90 = _percentile(times, 90) * 1000
        p99 = _percentile(times, 99) * 1000
        mean = statistics.mean(times) * 1000
        print(
            f"{h:>5d}x{w:<5d}      "
            f"{p50:>6.1f}ms  {p90:>6.1f}ms  {p99:>6.1f}ms  {mean:>6.1f}ms  "
            f"{peak_mb:>7.1f}"
        )


if __name__ == "__main__":
    main()
