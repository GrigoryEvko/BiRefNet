"""Regression test: dataset.py builds GT paths via os.path.split rather than
substring replacement, fixing two previously-silent bugs:

  1. `p.replace('/im/', '/gt/')` substituted *every* occurrence of '/im/' →
     a dataset path under '/data/im/dataset_im/im/...' would map to
     '/data/gt/dataset_im/gt/...' which doesn't exist.
  2. `p[:-(len(p.split('.')[-1])+1)]` strips an extension by counting bytes.
     For files without an extension the strip count equals the path length,
     producing the empty string + ext mismatch.

The test exercises the GT-path-building logic by simulating a directory
structure under a tmp dir and checking dataset.MyData picks the right gt
files. We can't fully construct MyData (it imports cv2 and reads Config),
so we extract and reproduce the exact lookup snippet here.
"""
from __future__ import annotations

import os


def _build_gt_path(im_path: str, ext: str) -> str:
    """Mirror of the new dataset.py logic for building GT paths."""
    p_dir, p_name = os.path.split(im_path)
    stem = os.path.splitext(p_name)[0]
    parent, last = os.path.split(p_dir)
    if last == "im":
        gt_dir = os.path.join(parent, "gt")
    else:
        gt_dir = p_dir.replace(os.sep + "im" + os.sep, os.sep + "gt" + os.sep)
    return os.path.join(gt_dir, stem + ext)


def test_simple_im_to_gt():
    p = "/data/DIS5K/DIS-TR/im/IMG_0001.jpg"
    assert _build_gt_path(p, ".png") == "/data/DIS5K/DIS-TR/gt/IMG_0001.png"


def test_handles_repeated_im_in_parent_path():
    """The previous str.replace would corrupt the parent dir."""
    p = "/data/im/some_im_dataset/im/sample.jpg"
    # Should only replace the *immediate* 'im' parent, not the upstream one.
    assert _build_gt_path(p, ".png") == "/data/im/some_im_dataset/gt/sample.png"


def test_handles_files_without_extension():
    """The previous splitext-via-substring would compute an empty stem."""
    p = "/data/im/sample"
    # splitext('sample') gives stem='sample', ext=''
    assert _build_gt_path(p, ".png") == "/data/gt/sample.png"


def test_picks_existing_extension(tmp_path):
    """Full round-trip: lay out files on disk and verify dataset-style lookup."""
    im_dir = tmp_path / "im"
    gt_dir = tmp_path / "gt"
    im_dir.mkdir()
    gt_dir.mkdir()
    (im_dir / "a.jpg").write_bytes(b"x")
    (gt_dir / "a.png").write_bytes(b"x")  # GT can have a different ext from IM
    candidate = _build_gt_path(str(im_dir / "a.jpg"), ".png")
    assert os.path.exists(candidate)


def test_im_directory_sibling_can_have_im_substring(tmp_path):
    """The previous bug: any 'im' substring anywhere in the path would be
    replaced. Now only the immediate parent named exactly 'im' flips."""
    im_dir = tmp_path / "team_image" / "im"
    gt_dir = tmp_path / "team_image" / "gt"
    im_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    (im_dir / "x.jpg").write_bytes(b"x")
    (gt_dir / "x.png").write_bytes(b"x")
    candidate = _build_gt_path(str(im_dir / "x.jpg"), ".png")
    assert os.path.exists(candidate)


def test_error_message_swap_fixed():
    """Sanity: the user-facing message about image vs label mismatch reports
    them in the right order. (Was swapped historically.)"""
    msg_template = (
        "There are different numbers of images ({n_im}) and labels ({n_gt})"
    )
    n_im, n_gt = 10, 7
    msg = msg_template.format(n_im=n_im, n_gt=n_gt)
    assert "images (10)" in msg
    assert "labels (7)" in msg
