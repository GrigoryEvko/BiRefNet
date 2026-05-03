import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    """Load an image from disk and return it as a PIL Image at the given size.

    cv2.imread silently returns None on read failure (missing file, permission
    error, unsupported format); cv2.resize would then crash with a confusing
    error. Validate the read and raise a FileNotFoundError that names the path.
    """
    ct = color_type.lower()
    if ct == 'rgb':
        image = cv2.imread(path)
    elif ct == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"color_type must be 'rgb' or 'gray', got {color_type!r}")
    if image is None:
        raise FileNotFoundError(f"cv2.imread returned None for {path!r} — file missing, unreadable, or unsupported format")
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if ct == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image



def check_state_dict(state_dict, unwanted_prefixes=('module.', '_orig_mod.')):
    """Strip wrapper prefixes that DDP / torch.compile add to state-dict keys.

    Repeatedly strips any matching prefix until none remains, so nested
    combinations like '_orig_mod.module.layer' (compile + DDP) collapse to
    'layer'. Raises if stripping would silently overwrite an existing key.
    """
    if not unwanted_prefixes:
        return state_dict
    prefixes = tuple(unwanted_prefixes)
    new_state_dict = {}
    for k in list(state_dict.keys()):
        new_k = k
        # strip while any prefix still matches; longest-first so order is irrelevant
        while True:
            stripped = False
            for p in sorted(prefixes, key=len, reverse=True):
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    stripped = True
                    break
            if not stripped:
                break
        if new_k in new_state_dict:
            raise ValueError(
                f"check_state_dict: prefix-strip collision — both {k!r} and "
                f"another key map to {new_k!r}; refusing to silently drop a value."
            )
        new_state_dict[new_k] = state_dict[k]
    state_dict.clear()
    state_dict.update(new_state_dict)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed, deterministic=False):
    """Seed all RNGs.

    `deterministic=True` opts into stricter determinism (slower; ~10-30% on
    HR training):
      - cudnn.deterministic=True + cudnn.benchmark=False
      - torch.use_deterministic_algorithms(True, warn_only=True)
      - CUBLAS_WORKSPACE_CONFIG=:4096:8 (required for cuBLAS determinism on
        recent CUDA; setting after the first cuBLAS call is too late so we
        also call os.environ.setdefault)
    Off by default: cudnn picks the fastest algo and TF32 is enabled (HR-friendly).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, AttributeError):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
