import pytest

from birefnet_api.buckets import (
    aspect_bucket,
    fit_into_bucket,
    nearest_bucket,
    round_to_multiple,
)


# --- round_to_multiple ----------------------------------------------------

def test_round_to_multiple_basic():
    assert round_to_multiple(0, 32) == 32        # min clamped
    assert round_to_multiple(15, 32) == 32       # rounds away from 0
    assert round_to_multiple(16, 32) == 32       # midpoint rounds up
    assert round_to_multiple(48, 32) == 64       # midpoint rounds up
    assert round_to_multiple(63, 32) == 64
    assert round_to_multiple(1024, 32) == 1024
    assert round_to_multiple(1025, 32) == 1024   # closer down


def test_round_to_multiple_rejects_nonpositive():
    with pytest.raises(ValueError):
        round_to_multiple(100, 0)
    with pytest.raises(ValueError):
        round_to_multiple(100, -8)


# --- aspect_bucket --------------------------------------------------------

def test_aspect_bucket_clamps_long_edge_landscape():
    h, w = aspect_bucket((8000, 12000), max_edge=2048, multiple=32)
    assert max(h, w) == 2048
    assert h % 32 == 0 and w % 32 == 0
    # aspect within +/-1 multiple
    expected_h = 2048 * 8000 / 12000
    assert abs(h - expected_h) <= 32


def test_aspect_bucket_clamps_long_edge_portrait():
    h, w = aspect_bucket((12000, 8000), max_edge=2048, multiple=32)
    assert max(h, w) == 2048
    assert h % 32 == 0 and w % 32 == 0
    expected_w = 2048 * 8000 / 12000
    assert abs(w - expected_w) <= 32


def test_aspect_bucket_no_upscale():
    """Inputs already <= max_edge stay roughly the same size after rounding."""
    h, w = aspect_bucket((900, 1280), max_edge=2048, multiple=32)
    assert h % 32 == 0 and w % 32 == 0
    assert h == round_to_multiple(900, 32)
    assert w == round_to_multiple(1280, 32)


def test_aspect_bucket_min_edge():
    # 12000 x 1 → max_edge 2048 forces width to be 2048 / 12000 ≈ 0.17
    # That should still produce a min_edge floor.
    h, w = aspect_bucket((12000, 1), max_edge=2048, multiple=32, min_edge=64)
    assert max(h, w) == 2048
    assert min(h, w) >= 64


def test_aspect_bucket_square():
    h, w = aspect_bucket((1024, 1024), max_edge=2048, multiple=32)
    assert h == w == 1024


def test_aspect_bucket_rejects_nonpositive():
    with pytest.raises(ValueError):
        aspect_bucket((0, 1024))
    with pytest.raises(ValueError):
        aspect_bucket((1024, -1))


def test_aspect_bucket_extreme_downscale_12k():
    """The user's scenario: 12K input scaled to 2K bucket."""
    h, w = aspect_bucket((8000, 12000), max_edge=2048, multiple=32)
    assert w == 2048
    # area shrinks by ~36x
    orig_area = 8000 * 12000
    bucket_area = h * w
    assert bucket_area < orig_area / 30


# --- nearest_bucket -------------------------------------------------------

def test_nearest_bucket_picks_closest_aspect():
    buckets = [(1024, 1024), (1024, 1536), (1536, 1024), (2048, 1024), (1024, 2048)]
    # 16:9 widescreen → 1024x2048 (closest aspect 2:1)
    pick = nearest_bucket((1080, 1920), buckets)
    # 1920/1080 ~= 1.78. The closest in the set is 1024x2048 (ar=2.0). Square is 1.0.
    assert pick == (1024, 2048)


def test_nearest_bucket_rejects_empty():
    with pytest.raises(ValueError):
        nearest_bucket((100, 100), [])


# --- fit_into_bucket ------------------------------------------------------

def test_fit_into_bucket_landscape_into_square():
    (rh, rw), (pt, pb, pl, pr) = fit_into_bucket((1080, 1920), (1024, 1024))
    # scale = min(1024/1080, 1024/1920) = 1024/1920 = 0.5333
    assert rw == 1024
    assert rh == round(1080 * 1024 / 1920)
    # vertical padding splits ~evenly
    assert pt + pb == 1024 - rh
    assert pl + pr == 0


def test_fit_into_bucket_already_fits():
    (rh, rw), pad = fit_into_bucket((512, 512), (1024, 1024))
    # scale = 2; resized goes 1024x1024
    assert rh == 1024 and rw == 1024
    assert pad == (0, 0, 0, 0)


def test_fit_into_bucket_preserves_aspect():
    (rh, rw), _ = fit_into_bucket((1080, 1920), (1024, 1024))
    # aspect ratio of resized matches original within 1px of rounding
    assert abs(rw / rh - 1920 / 1080) < 0.01
