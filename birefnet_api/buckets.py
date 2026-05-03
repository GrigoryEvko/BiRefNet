from __future__ import annotations

from typing import Sequence, Tuple


def round_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(multiple, ((int(x) + multiple // 2) // multiple) * multiple)


def aspect_bucket(
    orig_hw: Tuple[int, int],
    max_edge: int = 2048,
    multiple: int = 32,
    min_edge: int = 64,
) -> Tuple[int, int]:
    """Aspect-preserving bucket snapped to `multiple`.

    The longest side is clamped to `max_edge`; the shorter side is scaled
    proportionally and rounded to the nearest multiple. Both sides are at
    least `min_edge`.
    """
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"non-positive size {orig_hw}")
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        h = h * scale
        w = w * scale
    bh = max(min_edge, round_to_multiple(int(round(h)), multiple))
    bw = max(min_edge, round_to_multiple(int(round(w)), multiple))
    return bh, bw


def nearest_bucket(
    orig_hw: Tuple[int, int],
    buckets: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """Pick the bucket from `buckets` whose aspect ratio is closest to the input.

    Useful when you want to keep `torch.compile` cache hits hot by snapping every
    input to one of a small fixed set of (h, w) pairs. The image will be resized
    to the chosen bucket; preserve aspect by also letterboxing if needed (the
    predictor does this).
    """
    if not buckets:
        raise ValueError("buckets must be non-empty")
    h, w = int(orig_hw[0]), int(orig_hw[1])
    ar = w / max(1, h)
    return min(buckets, key=lambda b: abs((b[1] / max(1, b[0])) - ar))


def fit_into_bucket(
    orig_hw: Tuple[int, int],
    bucket_hw: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """Compute the resize and pad parameters to fit `orig_hw` into `bucket_hw`
    while preserving aspect ratio.

    Returns (resized_hw, (pad_top, pad_bottom, pad_left, pad_right)).
    """
    oh, ow = orig_hw
    bh, bw = bucket_hw
    scale = min(bh / oh, bw / ow)
    rh = max(1, int(round(oh * scale)))
    rw = max(1, int(round(ow * scale)))
    pad_v = bh - rh
    pad_h = bw - rw
    pad_t = pad_v // 2
    pad_b = pad_v - pad_t
    pad_l = pad_h // 2
    pad_r = pad_h - pad_l
    return (rh, rw), (pad_t, pad_b, pad_l, pad_r)
