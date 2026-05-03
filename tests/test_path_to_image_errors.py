"""Regression tests for path_to_image error handling.

cv2.imread returns None silently on missing/unreadable/unsupported files;
the next cv2.resize call would crash with an unhelpful "src is not a numpy
array" error. We now raise FileNotFoundError that names the path.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")


def test_missing_path_raises_file_not_found(tmp_path):
    from utils import path_to_image
    nonexistent = tmp_path / "no_such_file.png"
    with pytest.raises(FileNotFoundError, match="returned None"):
        path_to_image(str(nonexistent))


def test_invalid_color_type_raises_value_error(tmp_path):
    from utils import path_to_image
    p = tmp_path / "tmp.png"
    from PIL import Image
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(p)
    with pytest.raises(ValueError, match="color_type"):
        path_to_image(str(p), color_type="cmyk")


def test_path_to_image_works_for_rgb(tmp_path):
    """Sanity: working path still returns a PIL.Image (RGB)."""
    from PIL import Image
    from utils import path_to_image
    p = tmp_path / "img.png"
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(p)
    out = path_to_image(str(p), size=(16, 16), color_type="rgb")
    assert out.mode == "RGB"
    assert out.size == (16, 16)


def test_path_to_image_works_for_gray(tmp_path):
    from PIL import Image
    from utils import path_to_image
    p = tmp_path / "img.png"
    Image.new("L", (32, 32), color=128).save(p)
    out = path_to_image(str(p), size=(16, 16), color_type="gray")
    assert out.mode == "L"
    assert out.size == (16, 16)
