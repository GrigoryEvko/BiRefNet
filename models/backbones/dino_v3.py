"""DINOv3 backbone factories.

Each factory builds the timm DINOv3 model with the per-arch out_indices
preset. The factory accepts **kwargs that are forwarded to timm.create_model
— callers who know their input size is fixed should pass
`dynamic_img_size=False` for a measurable speedup (timm skips the
position-embedding interpolation on every forward).
"""
import timm


vit_model_to_out_indices = {
    'vit_small_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_small_plus_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_base_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_large_patch16_dinov3.lvd1689m': (5, 11, 17, 23),
    'vit_huge_plus_patch16_dinov3.lvd1689m': (7, 15, 23, 31),
    'vit_7b_patch16_dinov3.lvd1689m': (9, 19, 29, 39),
}


def _build(model_name: str, **overrides):
    """Common factory body. dynamic_img_size defaults to True (back-compat
    with the prior hardcoded value) but can be flipped via overrides."""
    out_indices = vit_model_to_out_indices[model_name]
    kwargs = dict(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    kwargs.update(overrides)
    return timm.create_model(**kwargs)


def dino_v3_s(**kwargs):
    return _build('vit_small_patch16_dinov3.lvd1689m', **kwargs)


def dino_v3_s_plus(**kwargs):
    return _build('vit_small_plus_patch16_dinov3.lvd1689m', **kwargs)


def dino_v3_b(**kwargs):
    return _build('vit_base_patch16_dinov3.lvd1689m', **kwargs)


def dino_v3_l(**kwargs):
    return _build('vit_large_patch16_dinov3.lvd1689m', **kwargs)


def dino_v3_h_plus(**kwargs):
    return _build('vit_huge_plus_patch16_dinov3.lvd1689m', **kwargs)


def dino_v3_7b(**kwargs):
    return _build('vit_7b_patch16_dinov3.lvd1689m', **kwargs)
