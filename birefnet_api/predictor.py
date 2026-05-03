from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from birefnet_api.buckets import aspect_bucket, fit_into_bucket, nearest_bucket

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor, str, "os.PathLike[str]"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _load_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        # Use a context manager so the file handle is released; .copy()
        # detaches the pixel data so we can keep using it after the file closes.
        with Image.open(os.fspath(image)) as fp:
            fp.load()
            return fp.copy()
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
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
        # CHW → HWC. Heuristic: if dim 0 looks like channels (1/3/4) AND the
        # last dim does NOT, permute. Symmetric ambiguous cases (3×3×H) bias
        # toward CHW because deep-learning code overwhelmingly produces that.
        if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
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
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device, non_blocking=True)
    return t.float().div_(255.0)


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
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype, None] = "bf16",
        max_edge: int = 2048,
        multiple: int = 32,
        normalize: bool = True,
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        channels_last: bool = True,
        buckets: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.device = torch.device(device)
        self.amp_dtype = self._resolve_dtype(dtype)
        self.max_edge = int(max_edge)
        self.multiple = int(multiple)
        self.normalize = bool(normalize)
        self.channels_last = bool(channels_last) and self.device.type == "cuda"
        self.buckets: Optional[List[Tuple[int, int]]] = (
            [(int(b[0]), int(b[1])) for b in buckets] if buckets else None
        )

        model = model.eval().to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        for p in model.parameters():
            p.requires_grad_(False)
        if compile:
            model = torch.compile(model, mode=compile_mode)
        self.model = model

        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    # --- factories ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str = "ZhengPeng7/BiRefNet", **kwargs) -> "BiRefNetPredictor":
        from models.birefnet import BiRefNet
        model = BiRefNet.from_pretrained(repo_id)
        return cls(model, **kwargs)

    @classmethod
    def from_checkpoint(cls, ckpt_path: Union[str, "os.PathLike[str]"], **kwargs) -> "BiRefNetPredictor":
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        model = BiRefNet(bb_pretrained=False)
        sd = torch.load(os.fspath(ckpt_path), map_location="cpu", weights_only=True)
        sd = check_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            # surface mismatches but don't fail outright
            print(f"[BiRefNetPredictor] state-dict: missing={len(missing)} unexpected={len(unexpected)}")
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
        pil = _load_pil(image)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        orig_w, orig_h = pil.size  # PIL is (w, h)
        bucket_hw = self._pick_bucket((orig_h, orig_w), max_edge)

        bucket_pil = pil if pil.size == (bucket_hw[1], bucket_hw[0]) else pil.resize(
            (bucket_hw[1], bucket_hw[0]), _LANCZOS
        )
        x = _pil_to_rgb_tensor(bucket_pil, self.device).unsqueeze(0)
        if self.normalize:
            x = (x - self._mean) / self._std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.amp_dtype is not None and self.device.type == "cuda":
            x = x.to(self.amp_dtype)

        mask = self._forward(x)  # [1,1,h,w] in autocast dtype (bf16 or fp32)
        # Upcast to fp32 BEFORE the upsample. The bf16 path saves time on the
        # tiny pre-upsample tensor but the post-upsample tensor is large at HR
        # (12K → 384MB fp32). Casting AFTER upsample meant bf16 mask + fp32
        # mask co-existed transiently → 576MB peak vs 384MB old. Casting first
        # keeps peak at the old level; the speed cost is on a small tensor.
        mask = mask.float()
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
            pil = _load_pil(img)
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
            resized = pil.resize((rw, rh), _LANCZOS)
            t = _pil_to_rgb_tensor(resized, self.device)
            if (pt, pb, pl, pr) != (0, 0, 0, 0):
                # F.pad takes (left, right, top, bottom)
                t = F.pad(t.unsqueeze(0), (pl, pr, pt, pb), mode="replicate").squeeze(0)
            tensors.append(t)
            pads.append((pt, pb, pl, pr, rh, rw))
        x = torch.stack(tensors, dim=0)
        if self.normalize:
            x = (x - self._mean) / self._std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.amp_dtype is not None and self.device.type == "cuda":
            x = x.to(self.amp_dtype)

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
            top = int(round(pt * m_h / bh))
            bot_off = int(round(pb * m_h / bh))
            left = int(round(pl * m_w / bw))
            right_off = int(round(pr * m_w / bw))
            m = masks[
                i : i + 1, :,
                top : (m_h - bot_off) if bot_off else m_h,
                left : (m_w - right_off) if right_off else m_w,
            ]
            # Upcast to fp32 before the upsample so peak memory equals
            # post-upsample fp32 only, not bf16 + fp32 simultaneously.
            m = F.interpolate(
                m.float(), size=(oh, ow), mode="bicubic", align_corners=False, antialias=True
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
        pil = _load_pil(image)
        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGB")
        rgb_pil = pil.convert("RGB")
        mask_t = cast("torch.Tensor", self.predict(rgb_pil, max_edge=max_edge, return_pil=False))
        rgb_np = np.asarray(rgb_pil, dtype=np.uint8)
        alpha_np = (mask_t.float().cpu().numpy() * 255.0).round().astype(np.uint8)
        if refine_fg:
            rgb_np = self._refine_foreground_np(rgb_np, alpha_np, r=fg_radius)
        rgba = np.dstack([rgb_np, alpha_np])
        return Image.fromarray(rgba, mode="RGBA")

    # --- internals ---------------------------------------------------------

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
            return Counter(picks).most_common(1)[0][0]
        per_item = [aspect_bucket(hw, max_edge=max_edge, multiple=self.multiple) for hw in items_hw]
        # Take the per-axis max so every image can be letterbox'd into it
        # without upscaling. (Aspect distortion is avoided by the letterbox.)
        bh = max(b[0] for b in per_item)
        bw = max(b[1] for b in per_item)
        return (bh, bw)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = (
            torch.amp.autocast("cuda", dtype=self.amp_dtype)
            if (self.amp_dtype is not None and self.device.type == "cuda")
            else nullcontext()
        )
        with ctx:
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                logits = out[-1]
            else:
                logits = out
            # sigmoid stays in autocast dtype (bf16 if amp on); the upsample
            # happens in the same dtype to halve interpolate memory at HR.
            # The fp32 cast moves to the predict() boundary right before
            # the user receives the mask.
            return logits.sigmoid()

    def _refine_foreground_np(self, rgb: np.ndarray, alpha: np.ndarray, r: int = 90) -> np.ndarray:
        """Photoroom-style two-pass blur fusion foreground estimation.

        Inputs are uint8 HWC and HW respectively. Output is uint8 HWC at the
        same resolution. Picks GPU vs CPU automatically based on free VRAM —
        the GPU path otherwise OOMs on multi-megapixel inputs (a 12K image
        needs ~5 GB transient float buffers).
        """
        h, w = alpha.shape
        r = max(1, min(int(r), max(1, min(h, w) - 1)))
        # ~7 fp32 buffers of (h, w, 3) survive simultaneously through the
        # two _fb_blur_pass calls. Be conservative.
        est_bytes = h * w * 3 * 4 * 8
        device = self._pick_refine_device(est_bytes)
        rgb_arr = np.array(rgb, copy=True)
        alpha_arr = np.array(alpha, copy=True)
        rgb_t = torch.from_numpy(rgb_arr).to(device).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        alpha_t = torch.from_numpy(alpha_arr).to(device).float().div_(255.0).unsqueeze(0).unsqueeze(0)
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
