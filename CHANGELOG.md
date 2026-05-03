# CHANGELOG

This fork of BiRefNet was hardened for production inference on torch 2.11+ /
CUDA 13.x. The HEAD set of changes is documented below grouped by impact
class. Anything marked **BREAKING** changes existing user-visible behavior;
read those before pulling into an existing deployment.

**Verified end-to-end against the real upstream `ZhengPeng7/BiRefNet`
checkpoint** (downloaded from HF Hub, run on a fresh deploy, no local
backbone weights file present). All four pretrained-smoke tests pass,
including the disk-image foreground/background separation > 0.1 check
that confirms `align_corners=True` default produces semantically correct
masks. The 4K HR-input test passes too. See
`tests/test_pretrained_smoke.py`.

The remaining production-hardware-only verification: bf16 + CUDA
deform_conv2d kernel availability. The CPU-only torchvision wheel on
this dev box lacks the CUDA deform_conv2d kernel; the user's k8s images
ship a full torchvision so the bf16 path will exercise on first L40S
boot. Run `BIREFNET_TEST_PRETRAINED=1 pytest tests/test_pretrained_smoke.py`
on the L40S to close this loop.

## Unreleased

### BREAKING

- **`BiRefNetPredictor` default device is now `"auto"`** (was `"cuda"`).
  Picks CUDA → MPS → CPU at instantiation time. Code that explicitly
  passed `device="cuda"` is unaffected. Code that relied on the implicit
  default and ran somewhere CUDA wasn't initialized at import will now
  silently land on CPU rather than raising. To restore old behavior:
  pass `device="cuda"` explicitly.

- **`BiRefNetPredictor.from_checkpoint(strict=True)` is now the default**
  (was implicit `strict=False`). A state-dict mismatch raises
  `RuntimeError` naming the offending keys instead of silently using
  random weights for the missing parameters. Same applies to
  `from_state_dict`. Pass `strict=False` explicitly to tolerate
  mismatches; in that mode the log lists the names instead of just
  counts.

- **`config.compile_mode` defaults to `"max-autotune-no-cudagraphs"`**
  (was `"default"`). First training run will spend 1–2 minutes on
  Triton kernel autotuning that wasn't there before — looks like a
  hang but isn't. Set `config.compile_mode = "default"` for the old
  fast-startup behavior.

- **`config.align_corners` exists, defaults to `True`** for compat with
  upstream pretrained weights. The earlier sweep that set every
  `align_corners=False` shifted predicted masks by ~½ pixel; this
  flag restores upstream-matching geometry. Set
  `config.align_corners = False` if your fork was retrained with the
  False-everywhere sweep.

- **`PixLoss` `loss_dict` now reports the **sum** across scales**, not
  the per-scale mean. The total loss tensor was always the sum; the
  dict log was off by a factor of N (number of scales). Existing
  wandb / tensorboard panels will see a step change in absolute values;
  ratios and trends are unchanged.

- **`predictor.from_checkpoint` and `from_state_dict`** now use
  `logging.getLogger("birefnet_api.predictor")` instead of `print()`
  for state-dict mismatch warnings. Default: silent. To see the
  warnings in production: configure `logging.basicConfig(level=INFO)`
  or set the predictor logger level explicitly.

- **`BiRefNetPredictor` enforces `max_pixels=200_000_000` by default**
  (decompression-bomb guard). Inputs larger than ~14k² are rejected
  before pixel-data decode. Pass `max_pixels=None` for trusted inputs
  with arbitrary dimensions, or pass an explicit larger cap.

### Added

- `BiRefNetPredictor.from_state_dict(sd, strict=True)` — build a
  predictor from an in-memory state dict, no temp file required.
- `BiRefNetPredictor.warmup(shapes=...)` — pre-drive a forward per
  shape so the first real request doesn't pay the
  compile/autotune/KV-allocation cost. Defaults to `self.buckets`
  when called with no args.
- `BiRefNetPredictor.__repr__` — single-line summary with backbone,
  param count, device, dtype, channels_last, buckets/max_edge.
- `device="auto"` — picks CUDA → MPS → CPU.
- OOM-aware `_forward`: catches `torch.cuda.OutOfMemoryError`, calls
  `empty_cache()`, and re-raises with the input shape, dtype, free /
  total VRAM, and remediation hints (lower max_edge, switch to bf16,
  split input).
- `tools/benchmark.py` — CLI for measuring p50/p90/p99 latency + peak
  VRAM at production sizes (1024² through 12K), against random-init or
  real upstream weights.

### Fixed (correctness)

- `BCELoss` → `BCEWithLogitsLoss` everywhere — `nn.BCELoss` raises a
  hard `RuntimeError` under bf16 autocast in torch >= 2.x.
- `StructureLoss` was getting double-sigmoid'd preds (PixLoss
  sigmoid'd everything before dispatching, but StructureLoss already
  applies sigmoid internally). Per-criterion `consumes_logits` flag
  routes raw logits to BCE/Structure and sigmoid'd preds to everyone
  else.
- `ContourLoss` `sqrt(0 + 1e-8)` underflows to `sqrt(0)` in bf16,
  whose backward emits `+inf`. Loss now upcasts to fp32 internally.
- `DeformableConv2d` falls back to fp32 under bf16 autocast — the
  CUDA kernel doesn't accept bf16 inputs.
- `F-measure` formula used `β` instead of `β²` per the standard
  saliency-detection formulation. Default `β=1` masks the bug; any
  non-default `β` gave wrong scores.
- `MBAMeasure.cal_ba` divided by zero on flat GTs (all-foreground or
  all-background masks have no boundary). Skip those radii; return 1.0
  if no radius produced any boundary pixels.
- `IoULoss` and `PatchIoULoss` vectorized — drop the per-batch Python
  loop. PatchIoULoss also had a dead `range(0, target.shape[0], 64)`
  loop iterating over batch+channel dims (only top-left 64×64 was
  scored) — now correctly tiles H/W.
- `image2patches` crashed when input dims weren't divisible by the
  grid size. Pads bottom-right with replicate to the next multiple.
- BiRefNet `mul_scl_ipt='add'` branch crashed on vgg/resnet backbones
  (called `self.bb(x)` instead of dispatching to the conv1..conv4
  attribute path). Now uses `_run_backbone` everywhere.
- `Decoder` `dec_ipt=False` branch hit a `NameError` because
  `dec_blk_in_channels` was only defined in the `dec_ipt=True` arm.
- `Decoder` `m4=None` (when `ms_supervision=False + out_ref=True +
  training=True`) crashed on `m4 * gdt_gt`.
- `freeze_bb=True` only set `requires_grad=False` on parameters; BN
  running stats are buffers and were still updating during training-mode
  forwards. `BiRefNet.train()` override now keeps the backbone in eval
  mode whenever `freeze_bb=True`.
- `Swin` SW-MSA attention mask used `float('-inf')` which saturates to
  the bf16 representable min; softmax over an all-min row produces NaN.
  Now uses `torch.finfo(dtype).min / 2`.
- `path_to_image` returned `None` silently on a bad path (cv2.imread
  fail); next cv2.resize crashed with a confusing "src is not a numpy
  array" error. Now raises `FileNotFoundError` naming the path.
- Train-time `set_seed` now fully drives `torch.use_deterministic_algorithms`
  and `CUBLAS_WORKSPACE_CONFIG` when `deterministic=True`.
- DDP-collate `dynamic_size` sampler dropped the spurious
  `tuple(sorted(...))` wrap that lex-swapped the W/H ranges when their
  bounds crossed.
- `_np_to_uint8` and the `_load_pil` tensor branch no longer collapse
  uniform-valued integer arrays to all-zero (a uint16 image of all-100
  used to come back all-0).
- Predictor `_pick_batch_bucket` no longer mixes max-H from one image
  with max-W from another. Letterbox-pads heterogeneous-aspect inputs
  into the smallest common bucket.
- Refine-foreground at HR no longer OOMs the GPU — picks GPU vs CPU
  device based on free VRAM and a budget heuristic.
- Eval-pipeline `cv2.imwrite` skeleton-cache write now uses tempfile +
  `os.replace` (atomic on POSIX) — the previous concurrent writes from
  the ThreadPoolExecutor could leave readers seeing partial files.
- Eval-pipeline `cv2.imread`-returns-`None` now logs the GT and pred
  paths plus the exception class instead of silently dropping the
  sample. Downstream the dataset shrinks but at least it's visible.
- `inference.py` reads label dimensions via `PIL.Image.open(...).size`
  (header-only) instead of `cv2.imread(...).shape` (full decode).
- `inference.py` model `.to(device)` hoisted out of the per-checkpoint
  loop. `torch.load` uses `map_location=device` to skip the CPU
  staging copy.
- `inference.py` `pin_memory` only when CUDA is actually available.
- DINO factories accept `**kwargs`; `build_backbone` passes
  `dynamic_img_size=False` automatically when `config.dynamic_size is None`.

### Performance

- Swin SW-MSA `attn_mask` cached per `(Hp, Wp, dtype, device)` with
  proper LRU eviction (was rebuilt every forward).
- Swin window-attention `relative_position_bias` cached per
  `(dtype, device)` in eval mode (rebuilt every forward in training).
  Cache cleared on `train()` and `_load_from_state_dict` to avoid
  stale views.
- Swin caches are **thread-safe**: per-instance `threading.Lock` guards
  the LRU update + eviction sequence so concurrent forwards in a
  threaded server (FastAPI run_in_threadpool, --workers, etc.) don't
  KeyError on `move_to_end` racing with `popitem`.
- `image_proc.mean_blur` is separable: O(k) per pixel, not O(k²).
  At r=90, ~45× fewer ops in the foreground-refinement path.
- Predictor mask path: `.float()` cast moved to BEFORE the bicubic
  upsample so peak memory is `output_size × 4 bytes`, not `output_size
  × 6 bytes` (bf16 mask + fp32 mask co-existing).

### Tests

- `tests/test_integration_real_model.py` — real BiRefNet end-to-end
  through the predictor (5 cases pass, 2 properly skip on no-CUDA /
  env-gated 12K).
- `tests/test_swin_cache_thread_safety.py` — 16-thread cache races,
  LRU under load, train/eval flip vs forward.
- `tests/test_predictor_decompression_bomb.py` — bomb-guard rejection
  paths for path / PIL / numpy / tensor inputs.
- `tests/test_predictor_soak.py` — leak detection: 100 sequential
  predicts grow bounded; predictor weakref dies after drop;
  construct/predict/drop loop releases models.
- `tests/test_cuda_bf16_smoke.py` — CUDA bf16 end-to-end (skipped
  without CUDA + Ampere+ tensor cores).
- `tests/test_pretrained_smoke.py` — opt-in (`BIREFNET_TEST_PRETRAINED=1`)
  download from HF Hub + sanity check on synthetic inputs. **The
  audit-identified production gate** — verifies state-dict prefix
  stripping, align_corners default, and end-to-end mask quality
  against real upstream weights.
- `tests/test_freeze_bb_bn_stats.py`, `test_align_corners_flag.py`
  (with F.interpolate monkey-patch capture), `test_iou_loss_vectorized.py`,
  `test_patch_iou_loss.py`, `test_contour_loss_bf16.py`,
  `test_mba_div_zero.py`, `test_path_to_image_errors.py`,
  `test_image2patches_pad.py`, `test_swin_attn_mask_cache.py`,
  `test_swin_rpb_cache.py`, `test_mean_blur_separable.py`,
  `test_predictor_extras.py`, plus the original
  `test_predictor_input_safety.py`, `test_loss_logit_dispatch.py`,
  `test_dynamic_size_sync.py`, `test_dataset_paths.py`,
  `test_buckets.py`, `test_predictor_api.py`, `test_load_weights.py`,
  `test_path_handling.py`, `test_decoder_shape.py`, `test_image_proc.py`,
  `test_fmeasure_formula.py`, `test_deform_conv_bf16.py`,
  `test_known_bugs.py`.

### Migration recipes

For an existing user pulling this fork into a deployment:

1. Pin the new `align_corners` default. Verify against your
   reference outputs at a representative input size. If your fork was
   trained with the `False`-everywhere sweep, set
   `config.align_corners = False` in `config.py`.
2. Audit `from_checkpoint` callers. Add `strict=False` if your
   serving code loads slightly-mismatched state dicts (legacy training
   scripts often did).
3. Set `max_pixels` explicitly at predictor construction. The default
   is generous (200 MP) but if you serve user uploads, document the
   upper bound your platform tolerates.
4. Configure `logging.getLogger("birefnet_api.predictor")` to capture
   the state-dict mismatch warnings. The previous `print()` lines
   ended up in stdout; the logger respects whatever handlers your
   serving framework has set up.
5. **Run `BIREFNET_TEST_PRETRAINED=1 pytest tests/test_pretrained_smoke.py`**
   on real hardware against your actual checkpoint — this is the
   audit-identified production gate.
