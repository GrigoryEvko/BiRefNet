from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from birefnet_api.buckets import aspect_bucket, nearest_bucket

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor, str, "os.PathLike[str]"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        return Image.open(os.fspath(image))
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, mode="L").convert("RGB")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(arr))
    if isinstance(image, torch.Tensor):
        t = image.detach()
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError("predict() expects a single image; use predict_batch() for batches")
            t = t.squeeze(0)
        if t.ndim != 3:
            raise ValueError(f"unsupported tensor shape {tuple(t.shape)}")
        if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        if t.shape[-1] == 4:
            t = t[..., :3]
        if t.shape[-1] == 1:
            t = t.expand(-1, -1, 3)
        if t.is_floating_point():
            arr = (t.clamp(0, 1).cpu() * 255.0).to(torch.uint8).numpy()
        else:
            arr = t.to(torch.uint8).cpu().numpy()
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

        mask = self._forward(x)  # [1,1,h,w] float32 in [0,1]
        mask = F.interpolate(
            mask, size=(orig_h, orig_w), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0.0, 1.0)
        mask = mask[0, 0]
        if return_pil:
            arr = (mask.detach().float().cpu().numpy() * 255.0).round().astype(np.uint8)
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

        tensors = []
        for pil, _ in items:
            resized = pil.resize((bucket_hw[1], bucket_hw[0]), _LANCZOS)
            tensors.append(_pil_to_rgb_tensor(resized, self.device))
        x = torch.stack(tensors, dim=0)
        if self.normalize:
            x = (x - self._mean) / self._std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.amp_dtype is not None and self.device.type == "cuda":
            x = x.to(self.amp_dtype)

        masks = self._forward(x)  # [B,1,h,w]
        out = []
        for i, (_, (oh, ow)) in enumerate(items):
            m = F.interpolate(
                masks[i : i + 1], size=(oh, ow), mode="bicubic", align_corners=False, antialias=True
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
        if self.buckets:
            from collections import Counter
            picks = [nearest_bucket(hw, self.buckets) for hw in items_hw]
            return Counter(picks).most_common(1)[0][0]
        max_h = max(hw[0] for hw in items_hw)
        max_w = max(hw[1] for hw in items_hw)
        return aspect_bucket((max_h, max_w), max_edge=max_edge, multiple=self.multiple)

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
        return logits.float().sigmoid()

    def _refine_foreground_np(self, rgb: np.ndarray, alpha: np.ndarray, r: int = 90) -> np.ndarray:
        """Photoroom-style two-pass blur fusion foreground estimation, on GPU.

        Inputs are uint8 HWC and HW respectively. Output is uint8 HWC at the same
        resolution. Falls back gracefully if `r >= min(H, W)`.
        """
        h, w = alpha.shape
        r = max(1, min(int(r), max(1, min(h, w) - 1)))
        device = self.device
        rgb_arr = np.array(rgb, copy=True)
        alpha_arr = np.array(alpha, copy=True)
        rgb_t = torch.from_numpy(rgb_arr).to(device).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        alpha_t = torch.from_numpy(alpha_arr).to(device).float().div_(255.0).unsqueeze(0).unsqueeze(0)
        fg, blur_b = _fb_blur_pass(rgb_t, rgb_t, rgb_t, alpha_t, r)
        fg, _ = _fb_blur_pass(rgb_t, fg, blur_b, alpha_t, max(1, min(6, r)))
        out = fg.clamp_(0.0, 1.0).mul_(255.0).round_().byte()
        return out[0].permute(1, 2, 0).contiguous().cpu().numpy()


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
