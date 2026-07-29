from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from birefnet_api.buckets import aspect_bucket, fit_into_bucket, nearest_bucket

# Default-OFF logger: applications that want predictor logs configure
# logging at the root or for "birefnet_api.predictor" specifically.
# Using getLogger means we don't interfere with pytest capture or with
# downstream applications that already have a logging setup.
_log = logging.getLogger(__name__)

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor, str, "os.PathLike[str]"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class NonFiniteMaskError(RuntimeError):
    """The model produced NaN or Inf in the predicted mask.

    Raised rather than returned because a non-finite mask is INVISIBLE
    downstream: NaN survives clamp(0, 1) and casts to 0 as uint8, producing a
    well-formed, fully transparent PNG that is indistinguishable from a
    legitimate "nothing detected" result. Callers that want to degrade
    gracefully should catch this explicitly and decide (retry on a different
    path, fall back to eager, return 503) — the one thing that must not happen
    is returning zeros as if they were an answer.
    """


def _np_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert an HWC or HW numpy array to uint8, auto-detecting common ranges.

    - uint8 → returned as-is
    - integer non-uint8 (uint16, int32, ...): rescaled by max-magnitude into
      [0, 255], not blindly truncated mod 256 (the original wrapper did the
      latter).
    - floating: detects [0,1] vs [0,255] vs general by inspecting the max
      value. Float in [0, 1.5] is treated as [0,1] (×255); anything above
      that is treated as already-[0,255] and just clipped.
    """
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        # Rescale into [0, 255] using observed range. For a uniform-valued
        # array (a_max == a_min) just clip-cast the raw value rather than
        # collapsing it to zero — a uint16 image of all-100 should stay
        # all-100 after the cast, not become all-0.
        a_min = float(arr.min())
        a_max = float(arr.max())
        if a_max <= a_min:
            return np.clip(arr, 0, 255).astype(np.uint8)
        return ((arr.astype(np.float64) - a_min) * (255.0 / (a_max - a_min))).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        a_max = float(np.nanmax(arr)) if arr.size else 0.0
        if a_max <= 1.5:
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)
    raise TypeError(f"unsupported numpy dtype: {arr.dtype}")


def _check_pixel_budget(h: int, w: int, max_pixels: Optional[int]) -> None:
    """Raise if the (h, w) area exceeds the budget — decompression-bomb guard."""
    if max_pixels is None:
        return
    area = int(h) * int(w)
    if area > max_pixels:
        raise ValueError(
            f"Input image dimensions {w}x{h} ({area:,} pixels) exceed "
            f"max_pixels={max_pixels:,}. This guard prevents decompression-bomb "
            f"attacks. Pass max_pixels=None at predictor construction (or a larger "
            f"cap) if the input is trusted."
        )


def _load_pil(image: ImageInput, max_pixels: Optional[int] = None) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        # Use a context manager so the file handle is released; .copy()
        # detaches the pixel data so we can keep using it after the file closes.
        # Image.open() is lazy — only the header is read until .load(), so
        # we can validate dimensions before paying the decompress cost.
        with Image.open(os.fspath(image)) as fp:
            w, h = fp.size
            _check_pixel_budget(h, w, max_pixels)
            fp.load()
            return fp.copy()
    if isinstance(image, Image.Image):
        w, h = image.size
        _check_pixel_budget(h, w, max_pixels)
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"unsupported numpy ndim={arr.ndim}, shape={arr.shape}; "
                f"expected (H, W) or (H, W, C). Pass a torch.Tensor for "
                f"batched/channels-first inputs."
            )
        _check_pixel_budget(arr.shape[0], arr.shape[1], max_pixels)
        if arr.ndim == 2:
            arr = _np_to_uint8(arr)
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = _np_to_uint8(arr)
        return Image.fromarray(np.ascontiguousarray(arr))
    if isinstance(image, torch.Tensor):
        t = image.detach()
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError("predict() expects a single image; use predict_batch() for batches")
            t = t.squeeze(0)
        if t.ndim != 3:
            raise ValueError(f"unsupported tensor shape {tuple(t.shape)}")
        # Defer pixel-budget check to after CHW->HWC normalization.
        # CHW → HWC. Heuristic: if dim 0 looks like channels (1/3/4) AND the
        # last dim does NOT, permute. Symmetric ambiguous cases (3×3×H) bias
        # toward CHW because deep-learning code overwhelmingly produces that.
        if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        # Now t is HWC — check budget before any conversion work.
        _check_pixel_budget(t.shape[0], t.shape[1], max_pixels)
        if t.shape[-1] == 4:
            t = t[..., :3]
        if t.shape[-1] == 1:
            t = t.expand(-1, -1, 3)
        if t.is_floating_point():
            t_max = float(t.max().item()) if t.numel() else 0.0
            if t_max <= 1.5:
                arr = (t.clamp(0, 1).cpu() * 255.0).to(torch.uint8).numpy()
            else:
                arr = t.clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            # Integer: rescale by observed range, don't truncate uint8 mod 256.
            # Uniform tensor (t_max == t_min) → clip-cast raw value, not zero.
            t = t.cpu()
            t_min = int(t.min().item())
            t_max = int(t.max().item())
            if t_max <= t_min:
                arr = t.clamp(0, 255).to(torch.uint8).numpy()
            else:
                f = (t.to(torch.float64) - t_min) * (255.0 / (t_max - t_min))
                arr = f.clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(np.ascontiguousarray(arr))
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _pil_to_rgb_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    # PIL's __array_interface__ returns a read-only view; torch.from_numpy
    # warns on read-only arrays. np.array(...) forces ONE CPU copy.
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).to(device, non_blocking=True)         # HWC uint8 GPU
    t = t.permute(2, 0, 1)                                          # CHW uint8 (non-contig view)
    # .to(float32) on a non-contig view produces a contiguous fp32 output
    # in a SINGLE memory pass (gather + cast). The previous explicit
    # .contiguous() before .to() did two passes (one to materialize uint8
    # contig, one to cast to fp32). At 12K HR: ~290MB GPU bandwidth saved.
    return t.to(torch.float32).div_(255.0)                          # CHW fp32 GPU contiguous


class BiRefNetPredictor:
    """High-level inference wrapper for arbitrary-resolution input.

    Workflow:
      1. Accept any size of input (PIL, np, torch, path).
      2. Snap to a model-friendly bucket (multiples of 32, longest edge <= max_edge).
      3. Downsample on CPU (cheap) or GPU, run the model in autocast.
      4. Upsample the mask back to the original resolution with bicubic+antialias.
      5. Return either a tensor mask or an RGBA cutout at the original resolution.

    The predictor never reads the project's `Config()` — it derives every setting
    from constructor arguments, so it does not require `train.sh` or the
    DIS5K-shaped directory layout to exist.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = "auto",
        dtype: Union[str, torch.dtype, None] = "bf16",
        max_edge: int = 2048,
        multiple: int = 32,
        normalize: bool = True,
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        channels_last: bool = True,
        buckets: Optional[Sequence[Tuple[int, int]]] = None,
        max_pixels: Optional[int] = 200_000_000,
    ):
        """
        max_pixels: decompression-bomb guard. Inputs whose H*W exceeds this
            cap are rejected before the pixel data is decoded. Default
            200_000_000 (~14000² or 12K×16K) — large enough for the user's
            12K cutout workflow, small enough to block a 50K malicious
            upload. Pass None to disable for trusted inputs only.
        """
        self.device = self._resolve_device(device)
        self.amp_dtype = self._resolve_dtype(dtype)
        self.max_edge = int(max_edge)
        self.multiple = int(multiple)
        self.normalize = bool(normalize)
        self.channels_last = bool(channels_last) and self.device.type == "cuda"
        self.buckets: Optional[List[Tuple[int, int]]] = (
            [(int(b[0]), int(b[1])) for b in buckets] if buckets else None
        )
        self.max_pixels = int(max_pixels) if max_pixels is not None else None

        model = model.eval().to(self.device)
        # Cast the entire model to amp_dtype (params + buffers) instead of
        # relying on torch.amp.autocast at forward time. Autocast keeps
        # BN/LN/etc. in fp32 internally, so under autocast the model's
        # *internal* tensor dtypes alternate between fp32 (after BN/LN) and
        # bf16 (after conv/linear). torch.compile/Dynamo guards on those
        # internal dtypes; with seven buckets and any fp32↔bf16 toggling on
        # different shapes, recompiles stack up and hit the per-function
        # cap of 8 — at which point Dynamo falls back to eager and the
        # cache we baked is partially wasted. Casting the model to a single
        # dtype gives a deterministic graph: one (shape, dtype) pair per
        # bucket, no autocast policy variability. fp16/bf16 BN works
        # natively in torch 2.x; the precision delta vs fp32-BN under
        # autocast is negligible at inference.
        if self.amp_dtype is not None:
            model = model.to(self.amp_dtype)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        for p in model.parameters():
            p.requires_grad_(False)
        if compile:
            # Swin's pyramid creates 4 stages with different (H, W) — with
            # 7 buckets that's >8 shape variants per inner function (window
            # partition, windowed attention, MLP). Default recompile cap of
            # 8 would force eager fallback for 9th+ variant; bump high
            # enough that all 7-bucket × 4-stage combinations compile
            # cleanly. force_parameter_static_shapes=False is what the
            # 'relative_position_bias_table size mismatch' warning
            # explicitly recommends — Swin's per-stage rpb table param has
            # different sizes per stage, and Dynamo defaults to guarding
            # parameters as static.
            try:
                # Empirical: 7-bucket Swin-L at HR triggers ≥80 unique shape
                # combinations on inner window-attention forward frames
                # (4 stages × varying window-flattened sizes × multiple
                # attention configs). Default cap of 8 forces eager fallback;
                # 64 still capped on window-attn forward in measurements.
                # 256 leaves enough headroom that no inner frame hits the
                # cap on this workload and the cache file we bake covers
                # every (bucket, stage, window) tuple cleanly.
                torch._dynamo.config.recompile_limit = max(
                    int(getattr(torch._dynamo.config, "recompile_limit", 8)),
                    256,
                )
                torch._dynamo.config.force_parameter_static_shapes = False
            except AttributeError:
                pass
            model = torch.compile(model, mode=compile_mode)
        self.model = model

        # Mean/std at amp_dtype so (x - mean) / std stays in target dtype
        # without broadcasting promotion. Saves a fp32 intermediate per
        # forward and removes a dtype-guard variable from Dynamo's view.
        buf_dtype = self.amp_dtype if self.amp_dtype is not None else torch.float32
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device, dtype=buf_dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device, dtype=buf_dtype).view(1, 3, 1, 1)

    # --- factories ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str = "ZhengPeng7/BiRefNet", **kwargs) -> "BiRefNetPredictor":
        """Download a published checkpoint from HuggingFace Hub and wrap it.

        bb_pretrained=False is forced because the HF state_dict already
        contains backbone weights — the default bb_pretrained=True would
        trigger a redundant load_weights() call that reads
        config.weights[bb_name] from the local filesystem and crashes with
        FileNotFoundError on fresh deployments where that file doesn't exist.
        """
        from models.birefnet import BiRefNet
        # Separate predictor-construction kwargs from BiRefNet-construction
        # kwargs. Anything not recognized as a BiRefNet arg flows through.
        model = BiRefNet.from_pretrained(repo_id, bb_pretrained=False)
        return cls(model, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, "os.PathLike[str]"],
        strict: bool = True,
        **kwargs,
    ) -> "BiRefNetPredictor":
        """Load a model from a local checkpoint and wrap it in a predictor.

        strict=True (default) raises on any missing/unexpected key after
        prefix stripping — a mismatched state_dict is almost always a config
        mismatch, and silently using random weights produces meaningless
        predictions. Pass strict=False to tolerate mismatches; in that mode
        the method names every mismatched key (up to 10 each direction)
        rather than just printing the counts.
        """
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        model = BiRefNet(bb_pretrained=False)
        sd = torch.load(os.fspath(ckpt_path), map_location="cpu", weights_only=True)
        sd = check_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            preview_n = 10
            miss_preview = ', '.join(missing[:preview_n]) + ('...' if len(missing) > preview_n else '')
            unx_preview = ', '.join(unexpected[:preview_n]) + ('...' if len(unexpected) > preview_n else '')
            msg = (
                f"state-dict mismatch loading {ckpt_path!s}: "
                f"missing={len(missing)} ({miss_preview or '-'}) "
                f"unexpected={len(unexpected)} ({unx_preview or '-'})"
            )
            if strict:
                raise RuntimeError(msg + " — pass strict=False to tolerate")
            _log.warning(msg)
        return cls(model, **kwargs)

    @classmethod
    def from_safetensors(
        cls,
        path: Union[str, "os.PathLike[str]"],
        strict: bool = True,
        **kwargs,
    ) -> "BiRefNetPredictor":
        """Load a model from a local .safetensors file (single shard).

        Skips HF Hub entirely — useful in air-gapped containers where
        HF_HUB_OFFLINE=1 blocks all network access.

        Memory-conscious load path (audited on Swin-L BiRefNet HR, 220M
        params, 423 MB FP16 safetensors → 852 MB FP32 model):

          1. Construct the model on torch.device('meta'): zero allocation.
          2. to_empty(target_device): allocates uninitialised FP32 params on
             the destination (GPU when CUDA available). 852 MB target alloc.
          3. safetensors.load_file(device=target): file's native dtype loads
             straight into target memory. +451 MB target peak (FP16 sd).
          4. load_state_dict(strict=False, assign=False): in-place copy_()
             with dtype conversion FP16→FP32 into the already-allocated
             params. No extra allocation; sd freed at del.

        vs. the legacy CPU→GPU staging path (fresh-process measurements):
          - CPU peak RSS: 1.57 GB → was 2.27 GB (saves ~700 MB host RAM)
          - GPU peak: 1.30 GB transient → was 852 MB (+450 MB peak)
          - Load wall time: 0.68s → was 1.74s (-61%, skips CPU random init
            + CPU→GPU PCIe copy of 880 MB)

        The kiosk pod is host-RAM-bound (multiple models in one process),
        so trading host MB for GPU MB is a clean win — L40S has 33 GB on
        the partition.

        Falls back to the legacy CPU-staging path if anything in the
        meta-device pipeline fails — some custom modules can't be
        constructed without real tensors and we want a clean degradation.
        """
        from safetensors.torch import load_file
        from models.birefnet import BiRefNet
        from utils import check_state_dict

        # Resolve target device the same way predictor __init__ does so the
        # weights land where the model will live.
        device_arg = kwargs.get("device", "auto")
        target_dev = cls._resolve_device(device_arg)

        try:
            with torch.device("meta"):
                model = BiRefNet(bb_pretrained=False)
            # to_empty allocates uninitialised storage on target_dev for every
            # parameter and buffer of the meta-built module (deep, not just
            # top-level). Result: model has FP32 params on GPU, garbage data.
            model = model.to_empty(device=target_dev)

            sd_device = "cuda" if target_dev.type == "cuda" else "cpu"
            sd = load_file(os.fspath(path), device=sd_device)
            sd = check_state_dict(sd)

            # assign=False: in-place copy_() preserves the FP32 storage from
            # to_empty and does dtype conversion FP16→FP32 during the copy.
            # assign=True would replace params with the FP16 sd tensors,
            # which corrupts BatchNorm forward (BN under autocast wants
            # FP32 weights — silently produces "Expected weight to have
            # type Float but got Half" at runtime).
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=False)
            if missing or unexpected:
                preview_n = 10
                miss_preview = ', '.join(missing[:preview_n]) + ('...' if len(missing) > preview_n else '')
                unx_preview = ', '.join(unexpected[:preview_n]) + ('...' if len(unexpected) > preview_n else '')
                msg = (
                    f"state-dict mismatch loading {path!s}: "
                    f"missing={len(missing)} ({miss_preview or '-'}) "
                    f"unexpected={len(unexpected)} ({unx_preview or '-'})"
                )
                if strict:
                    raise RuntimeError(msg + " — pass strict=False to tolerate")
                _log.warning(msg)
            del sd
            # The predictor's __init__ will call .to(device); already on
            # target_dev so it's a no-op (PyTorch fast-paths same-device .to).
            return cls(model, **kwargs)
        except (RuntimeError, NotImplementedError) as e:
            _log.warning(
                "meta-device load path failed (%s); falling back to CPU staging "
                "via from_state_dict. This uses ~1.6 GB more host RAM at peak.",
                e,
            )
            sd = load_file(os.fspath(path), device="cpu")
            return cls.from_state_dict(sd, strict=strict, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        strict: bool = True,
        **kwargs,
    ) -> "BiRefNetPredictor":
        """Build a predictor from an in-memory state_dict (dict of tensors).

        Useful when the caller already loaded the weights (e.g. from a
        deployment artifact, a model registry, or remote storage) and
        doesn't want a temp file on disk. Same prefix-stripping and
        strict-checking semantics as from_checkpoint.
        """
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        model = BiRefNet(bb_pretrained=False)
        sd = check_state_dict(dict(state_dict))  # copy: check_state_dict mutates
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            preview_n = 10
            miss_preview = ', '.join(missing[:preview_n]) + ('...' if len(missing) > preview_n else '')
            unx_preview = ', '.join(unexpected[:preview_n]) + ('...' if len(unexpected) > preview_n else '')
            msg = (
                f"state-dict mismatch: missing={len(missing)} ({miss_preview or '-'}) "
                f"unexpected={len(unexpected)} ({unx_preview or '-'})"
            )
            if strict:
                raise RuntimeError(msg + " — pass strict=False to tolerate")
            _log.warning(msg)
        return cls(model, **kwargs)

    # --- public api --------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        image: ImageInput,
        max_edge: Optional[int] = None,
        return_pil: bool = False,
    ) -> Union[torch.Tensor, Image.Image]:
        """Run inference on a single image and return the mask at original resolution.

        Returns a torch.Tensor of shape (H, W) in [0, 1] (default), or a PIL 'L' image.
        """
        max_edge = self.max_edge if max_edge is None else int(max_edge)
        pil = _load_pil(image, max_pixels=self.max_pixels)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        orig_w, orig_h = pil.size  # PIL is (w, h)
        bucket_hw = self._pick_bucket((orig_h, orig_w), max_edge)

        # Letterbox into the bucket so explicit-bucket users (constructor
        # buckets=...) don't get aspect-distorted masks. With aspect_bucket
        # this is effectively a no-op (rh, rw == bucket_hw). With nearest
        # bucket selection from a small fixed set, it's the difference
        # between a faithful prediction and a squashed one.
        (rh, rw), (pt, pb, pl, pr) = fit_into_bucket((orig_h, orig_w), bucket_hw)
        bucket_pil = pil if pil.size == (rw, rh) else pil.resize((rw, rh), _LANCZOS)
        x = _pil_to_rgb_tensor(bucket_pil, self.device).unsqueeze(0)
        if (pt, pb, pl, pr) != (0, 0, 0, 0):
            x = F.pad(x, (pl, pr, pt, pb), mode="replicate")
        x = self._finalize_input(x)

        mask = self._forward(x)  # [1,1,h,w] fp32 (sigmoid is now in fp32 in _forward)
        # Crop the letterbox padding from the mask before upsample.
        bh, bw = bucket_hw
        m_h, m_w = mask.shape[-2], mask.shape[-1]
        if (pt, pb, pl, pr) != (0, 0, 0, 0):
            top = max(0, int(round(pt * m_h / bh)))
            bot_off = max(0, int(round(pb * m_h / bh)))
            left = max(0, int(round(pl * m_w / bw)))
            right_off = max(0, int(round(pr * m_w / bw)))
            end_h = max(top + 1, m_h - bot_off)
            end_w = max(left + 1, m_w - right_off)
            mask = mask[..., top:end_h, left:end_w]
        mask = F.interpolate(
            mask, size=(orig_h, orig_w), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0.0, 1.0)
        mask = mask[0, 0]
        if return_pil:
            arr = (mask.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        return mask

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Iterable[ImageInput],
        max_edge: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Run inference on a list of images. Each result is upsampled to its own
        original resolution. Within a batch, all images are bucketed to the same
        (h, w) so a single forward call can serve heterogeneous-aspect inputs.
        """
        max_edge = self.max_edge if max_edge is None else int(max_edge)
        items = []
        for img in images:
            pil = _load_pil(img, max_pixels=self.max_pixels)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            ow, oh = pil.size
            items.append((pil, (oh, ow)))
        if not items:
            return []
        bucket_hw = self._pick_batch_bucket([item[1] for item in items], max_edge)

        # Letterbox-resize into the bucket so heterogeneous aspects stay sane:
        # the previous _pick_batch_bucket took (max_h, max_w) from independent
        # images and resized everyone to that aspect, distorting both ends.
        tensors = []
        pads = []
        for pil, (oh, ow) in items:
            (rh, rw), (pt, pb, pl, pr) = fit_into_bucket((oh, ow), bucket_hw)
            # Skip the LANCZOS pass when the image already matches the
            # bucket — saves ~50ms per 4K image when callers send
            # bucket-sized inputs.
            resized = pil if pil.size == (rw, rh) else pil.resize((rw, rh), _LANCZOS)
            t = _pil_to_rgb_tensor(resized, self.device)
            if (pt, pb, pl, pr) != (0, 0, 0, 0):
                # F.pad takes (left, right, top, bottom)
                t = F.pad(t.unsqueeze(0), (pl, pr, pt, pb), mode="replicate").squeeze(0)
            tensors.append(t)
            pads.append((pt, pb, pl, pr, rh, rw))
        x = torch.stack(tensors, dim=0)
        # The per-item tensors are no longer needed — drop the strong refs
        # so they can be GC'd while the model forward runs (saves B × C × H ×
        # W × 4 bytes of peak VRAM at HR batch).
        del tensors
        x = self._finalize_input(x)

        masks = self._forward(x)  # [B,1,bh,bw]
        # Crop the letterbox padding from each mask, then upsample to the
        # corresponding original resolution. The mask spatial size may not equal
        # bucket_hw exactly (decoder keeps it the same in BiRefNet), so we map
        # pad amounts proportionally.
        out = []
        bh, bw = bucket_hw
        m_h, m_w = masks.shape[-2], masks.shape[-1]
        for i, (_, (oh, ow)) in enumerate(items):
            pt, pb, pl, pr, _rh, _rw = pads[i]
            # Clamp + max(start+1, end) so a future model that returns a
            # different m_h vs bh (e.g. some torch.compile rounding) can't
            # produce an empty or backwards slice that would crash
            # F.interpolate.
            top = max(0, int(round(pt * m_h / bh)))
            bot_off = max(0, int(round(pb * m_h / bh)))
            left = max(0, int(round(pl * m_w / bw)))
            right_off = max(0, int(round(pr * m_w / bw)))
            end_h = max(top + 1, m_h - bot_off)
            end_w = max(left + 1, m_w - right_off)
            m = masks[i : i + 1, :, top:end_h, left:end_w]
            # Sigmoid is now in fp32 in _forward, so m is already fp32.
            m = F.interpolate(
                m, size=(oh, ow), mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0.0, 1.0)
            out.append(m[0, 0])
        return out

    @torch.inference_mode()
    def cutout(
        self,
        image: ImageInput,
        max_edge: Optional[int] = None,
        refine_fg: bool = False,
        fg_radius: int = 90,
    ) -> Image.Image:
        """Return an RGBA PIL image at the original resolution with mask as alpha."""
        pil = _load_pil(image, max_pixels=self.max_pixels)
        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGB")
        rgb_pil = pil.convert("RGB")
        mask_t = cast("torch.Tensor", self.predict(rgb_pil, max_edge=max_edge, return_pil=False))
        rgb_np = np.asarray(rgb_pil, dtype=np.uint8)
        # Refine reads the GPU mask tensor directly — the previous flow
        # CPU-quantized alpha to uint8 then re-uploaded it as fp32, an
        # unnecessary round-trip costing ~2× the mask size at HR.
        if refine_fg:
            rgb_np = self._refine_foreground_np(rgb_np, mask_t, r=fg_radius)
        # Quantize to uint8 on the GPU before transferring — the full-res
        # fp32 mask is 4 bytes/pixel; uint8 is 1. At 12K that's 96MB vs
        # 384MB across PCIe, ~11ms saved per cutout on a typical bus.
        alpha_np = mask_t.mul(255).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        rgba = np.dstack([rgb_np, alpha_np])
        return Image.fromarray(rgba, mode="RGBA")

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _resolve_device(device) -> torch.device:
        """Pick a torch.device, supporting 'auto' for best-available."""
        if isinstance(device, torch.device):
            return device
        if device != "auto":
            return torch.device(device)
        # 'auto': prefer CUDA, then MPS (Apple Silicon), then CPU.
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_avail = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)
        if mps_avail():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        d = str(dtype).lower()
        if d in ("fp32", "float32", "f32"):
            return None
        if d in ("fp16", "float16", "half"):
            return torch.float16
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        raise ValueError(f"unsupported dtype: {dtype!r}")

    def _pick_bucket(self, orig_hw: Tuple[int, int], max_edge: int) -> Tuple[int, int]:
        if self.buckets:
            return nearest_bucket(orig_hw, self.buckets)
        return aspect_bucket(orig_hw, max_edge=max_edge, multiple=self.multiple)

    def _pick_batch_bucket(self, items_hw: List[Tuple[int, int]], max_edge: int) -> Tuple[int, int]:
        """Pick a single (bucket_h, bucket_w) for the batch.

        With explicit buckets: snap to the most-common nearest bucket.
        Without explicit buckets: pick the largest aspect_bucket across the
        items, which is the smallest bucket every image fits inside (after
        letterbox padding) without upscaling beyond max_edge.
        """
        if self.buckets:
            from collections import Counter
            picks = [nearest_bucket(hw, self.buckets) for hw in items_hw]
            counts = Counter(picks)
            # Deterministic tiebreak: most-common, then smallest area, then
            # tuple ordering. Without this, Counter.most_common(1) returns
            # whichever insertion order the iteration produced — different
            # request ordering produces different bucket choice → torch.compile
            # cache thrash in production.
            return max(counts, key=lambda b: (counts[b], -b[0] * b[1], b))
        per_item = [aspect_bucket(hw, max_edge=max_edge, multiple=self.multiple) for hw in items_hw]
        # Per-axis max ensures every image can be letterbox'd in without
        # upscaling. But max-h from item A and max-w from item B can stack
        # into a square that's larger than max_edge × max_edge. Cap the long
        # axis so we don't accidentally allocate a 4MP square for a batch of
        # rectangles.
        bh = max(b[0] for b in per_item)
        bw = max(b[1] for b in per_item)
        long_axis = max(bh, bw)
        if long_axis > max_edge:
            scale = max_edge / long_axis
            m = self.multiple
            bh = max(m, ((int(bh * scale) + m // 2) // m) * m)
            bw = max(m, ((int(bw * scale) + m // 2) // m) * m)
        return (bh, bw)

    def _finalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Single normalisation + memory-format + dtype pipeline.

        Used by predict / predict_batch / warmup so they all build the same
        compiled graph. Order matters:

          1. Cast to amp_dtype FIRST so every downstream op runs in target
             dtype. Doing this last (the previous order) left fp32
             intermediates from `(x - mean) / std`, which under autocast
             cascade into different internal dtypes per bucket and
             explode Dynamo's recompile counter past its 8-per-function cap.
          2. (x - mean) / std with already-amp_dtype mean/std (set in
             __init__) so subtraction stays in target dtype with no
             broadcasting promotion.
          3. channels_last reformat last — channels_last is a memory
             layout, doesn't change dtype.
        """
        if self.amp_dtype is not None and x.dtype != self.amp_dtype:
            x = x.to(self.amp_dtype)
        if self.normalize:
            x = (x - self._mean) / self._std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # No autocast: model was already cast to amp_dtype in __init__, so
        # there's no fp32-vs-bf16 ambiguity for Dynamo to guard on.
        try:
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                logits = out[-1]
            else:
                logits = out
            # Sigmoid in fp32 (matches inference.py): bf16 sigmoid quantizes
            # values near 1.0 into ~5 bins (7-bit mantissa), losing useful
            # mask precision. The upcast cost is on a small pre-upsample
            # tensor — negligible vs preserving 23-bit precision.
            mask = logits.float().sigmoid()

            # REFUSE to return a non-finite mask.
            #
            # Without this the failure is completely silent and looks healthy.
            # A NaN mask survives clamp_(0, 1) — clamp propagates NaN, it does
            # not remove it — and then `(mask * 255).round().astype(np.uint8)`
            # turns every NaN into 0. The caller gets a well-formed PNG that is
            # uniformly transparent, HTTP 200, and /health still reports "ok".
            #
            # That is not hypothetical. eu-north-1 served exactly this for six
            # days: every /background/mask response min=0 max=0 mean=0, 100%
            # transparent, in 0.04s, with nothing in the logs but a NumPy
            # "invalid value encountered in cast" RuntimeWarning at the cast
            # site. Nothing upstream of that warning noticed, because a mask of
            # all zeros is a VALID mask — it just means "nothing here".
            #
            # NOTE this guard catches only the NON-FINITE failure, and the
            # incident that motivated it was NOT non-finite. The real defect was
            # a memory-layout mismatch: the AOTI bundle was exported on a
            # contiguous tensor while this class feeds channels_last, giving
            # finite-but-saturated output (min=196, 99.7% opaque, centre-corner
            # +3.6 vs eager's +129.0). Wrong and finite still gets served, and no
            # runtime assert on the OUTPUT can catch that. It is caught at the
            # input instead — aoti/bundle.py records input_strides and refuses a
            # mismatch (BundleLayoutMismatch). This guard remains for genuine
            # NaN, which is cheap to check and catastrophic to serve.
            #
            # The check is one reduction over the small pre-upsample tensor. It
            # forces a device sync, which is the price of not shipping silence.
            if not torch.isfinite(mask).all():
                n_bad = int((~torch.isfinite(mask)).sum())
                raise NonFiniteMaskError(
                    f"model produced {n_bad} non-finite values out of "
                    f"{mask.numel()} at input {tuple(x.shape)} dtype={x.dtype}. "
                    f"A NaN mask casts to an all-zero (fully transparent) PNG, "
                    f"so this is raised rather than returned."
                )
            return mask
        except torch.cuda.OutOfMemoryError as e:
            # Be defensive: empty_cache() / mem_get_info() can themselves
            # raise on a corrupted CUDA context (the rotation pattern your
            # k8s deploy uses every 24h is precisely to mitigate this).
            # Always preserve the original OOM as the cause.
            free = total = None
            try:
                torch.cuda.empty_cache()
                if self.device.type == "cuda":
                    free, total = torch.cuda.mem_get_info(self.device)
            except Exception:
                pass
            free_str = f"{free/1e9:.2f} GB" if free is not None else "unknown"
            total_str = f"{total/1e9:.2f} GB" if total is not None else "unknown"
            raise torch.cuda.OutOfMemoryError(
                f"OOM during forward at input {tuple(x.shape)}, dtype={x.dtype}, "
                f"device={self.device}. After empty_cache(): "
                f"{free_str} free / {total_str} total. "
                f"Try lowering max_edge (current bucket = {tuple(x.shape[-2:])}), "
                f"using dtype='bf16' on Ampere+, or splitting the input. "
                f"Original: {e}"
            ) from e

    @torch.inference_mode()
    def warmup(self, shapes: Optional[Sequence[Tuple[int, int]]] = None) -> None:
        """Pre-compile / pre-allocate for known input shapes.

        Drives one forward pass per (h, w) in `shapes` (defaults to the
        explicit `buckets` if any). Useful before serving begins so the
        first real request doesn't pay the torch.compile / cuDNN-autotune
        / KV-allocation cost.

        No-op if neither `shapes` nor `buckets` are provided.
        """
        targets = list(shapes or self.buckets or [])
        for hw in targets:
            h, w = int(hw[0]), int(hw[1])
            # Match predict()'s pipeline exactly: synthesize an fp32 tensor
            # then route through _finalize_input. This guarantees the
            # compiled graph the warmup populates is the SAME graph predict
            # will hit at request time — no second compile from
            # warmup-vs-predict divergence.
            x = torch.zeros(1, 3, h, w, device=self.device, dtype=torch.float32)
            x = self._finalize_input(x)
            self._forward(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __repr__(self) -> str:
        bb_name = type(getattr(self.model, "bb", self.model)).__name__
        n_params = sum(p.numel() for p in self.model.parameters())
        amp = (
            self.amp_dtype.__str__().rsplit(".", 1)[-1]
            if self.amp_dtype is not None
            else "fp32"
        )
        bucket_str = (
            f", buckets={len(self.buckets)}"
            if self.buckets is not None
            else f", max_edge={self.max_edge}"
        )
        return (
            f"BiRefNetPredictor(bb={bb_name}, params={n_params/1e6:.1f}M, "
            f"device={self.device}, dtype={amp}, "
            f"channels_last={self.channels_last}{bucket_str})"
        )

    def _refine_foreground_np(self, rgb: np.ndarray, alpha, r: int = 90) -> np.ndarray:
        """Photoroom-style two-pass blur fusion foreground estimation.

        Inputs:
          - rgb: uint8 HWC numpy
          - alpha: either uint8 HW numpy OR a torch.Tensor in [0, 1]
            (HW or 1×1×H×W or H×W shape). Passing a tensor lets cutout()
            skip a CPU round-trip of the mask.

        Output: uint8 HWC numpy at the same resolution. Picks GPU vs CPU
        automatically based on free VRAM.
        """
        if isinstance(alpha, torch.Tensor):
            # Mask was passed directly from predict() — already on the right
            # device, already fp32 in [0,1]. Use shape from the tensor.
            alpha_squeezed = alpha.squeeze() if alpha.ndim > 2 else alpha
            h, w = alpha_squeezed.shape[-2], alpha_squeezed.shape[-1]
        else:
            h, w = alpha.shape
        r = max(1, min(int(r), max(1, min(h, w) - 1)))
        # Two avg_pool2d separable passes × ~6 transient buffers each.
        # Bumped to 12 from 8 — at 12K input the previous 8x estimate
        # under-counted and led to GPU OOM on smaller GPUs.
        est_bytes = h * w * 3 * 4 * 12
        device = self._pick_refine_device(est_bytes)
        # Drop redundant np.array(copy=True) — caller already owns these
        # buffers, the torch.from_numpy view + .to(device) does the actual
        # copy onto the target device. At 12K: saves ~432MB CPU + 144MB.
        # PIL's array interface returns read-only buffers; torch.from_numpy
        # warns on those. .copy() forces a writable buffer; on already-writable
        # input we still pay one copy, but that's better than the silent
        # undefined-behavior path the warning calls out.
        if not rgb.flags.writeable:
            rgb = rgb.copy()
        rgb_t = torch.from_numpy(rgb).to(device).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        if isinstance(alpha, torch.Tensor):
            # Already a GPU fp32 mask; just shape it to (1, 1, H, W).
            alpha_t = alpha.to(device).float()
            while alpha_t.ndim < 4:
                alpha_t = alpha_t.unsqueeze(0)
        else:
            alpha_t = torch.from_numpy(alpha).to(device).float().div_(255.0).unsqueeze(0).unsqueeze(0)
        fg, blur_b = _fb_blur_pass(rgb_t, rgb_t, rgb_t, alpha_t, r)
        fg, _ = _fb_blur_pass(rgb_t, fg, blur_b, alpha_t, max(1, min(6, r)))
        out = fg.clamp_(0.0, 1.0).mul_(255.0).round_().byte()
        return out[0].permute(1, 2, 0).contiguous().cpu().numpy()

    def _pick_refine_device(self, est_bytes: int) -> torch.device:
        """Decide whether the foreground refine step fits on self.device.

        Falls back to CPU when self.device is CUDA and est_bytes exceeds 60%
        of free VRAM (leaves headroom for the model and other allocations).
        """
        if self.device.type != "cuda":
            return self.device
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return self.device
        if est_bytes > 0.6 * free:
            return torch.device("cpu")
        return self.device


def _box_blur(x: torch.Tensor, r: int) -> torch.Tensor:
    """Separable box filter approximating cv2.blur; preserves the input dtype."""
    if r % 2 == 0:
        pad_l, pad_t = r // 2 - 1, r // 2 - 1
        pad_r, pad_b = r // 2, r // 2
    else:
        pad_l = pad_r = pad_t = pad_b = r // 2
    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    # two-pass separable: (r,1) then (1,r). Same result as a single (r,r) avg pool
    # but O(r) instead of O(r^2) work.
    y = F.avg_pool2d(x_pad, kernel_size=(r, 1), stride=1, count_include_pad=False)
    y = F.avg_pool2d(y, kernel_size=(1, r), stride=1, count_include_pad=False)
    return y


def _fb_blur_pass(image, fg, b, alpha, r):
    blurred_alpha = _box_blur(alpha, r)
    blurred_fga = _box_blur(fg * alpha, r)
    blurred_fg = blurred_fga / (blurred_alpha + 1e-5)
    blurred_b1a = _box_blur(b * (1 - alpha), r)
    blurred_b = blurred_b1a / ((1 - blurred_alpha) + 1e-5)
    fg_out = blurred_fg + alpha * (image - alpha * blurred_fg - (1 - alpha) * blurred_b)
    return fg_out, blurred_b
