import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from contextlib import nullcontext

from dataset import MyData
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict
from config import Config


config = Config()

mixed_precision = config.mixed_precision
if mixed_precision == 'fp16':
    mixed_dtype = torch.float16
elif mixed_precision == 'bf16':
    mixed_dtype = torch.bfloat16
else:
    mixed_dtype = None

autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=mixed_dtype) if mixed_dtype else nullcontext()


def inference(model, data_loader_test, pred_root, method, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    for batch in tqdm(data_loader_test, total=len(data_loader_test)) if config.verbose_eval else data_loader_test:
        inputs = batch[0].to(device)
        label_paths = batch[-1]
        with autocast_ctx, torch.no_grad():
            logits = model(inputs)[-1]
        # Cast outside autocast so the sigmoid runs in fp32 — and we don't
        # waste a bf16 sigmoid that gets immediately upcast.
        scaled_preds = logits.float().sigmoid()

        os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            # PIL.Image.open() reads only the header to get .size, so we
            # avoid decoding the entire label PNG just for shape (the old
            # cv2.imread().shape pulled the full pixel data).
            with Image.open(label_paths[idx_sample]) as _label_pil:
                target_w, target_h = _label_pil.size
            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=bool(getattr(config, 'align_corners', True))
            )
            save_tensor_img(res, os.path.join(os.path.join(pred_root, method, testset), label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name
    if model_training:
        model.train()
    return None


def main(args):
    device = config.device
    if args.ckpt_folder:
        print('Testing with models in {}'.format(args.ckpt_folder))
    else:
        print('Testing with model {}'.format(args.ckpt))

    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
    else:
        print('Undefined model: {}.'.format(config.model))
        return None
    # Move the model to the target device once. The per-checkpoint loop
    # below only swaps weights via load_state_dict — the parameter tensors
    # stay on `device`, so re-calling `.to(device)` every iteration was a
    # no-op that still walked every parameter.
    model = model.to(device)
    def _epoch_of(path):
        # 'foo/epoch_42.pth' → 42. Tolerate stems whose last char is in
        # {'.', 'p', 't', 'h'} which the previous rstrip('.pth') corrupted.
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.split('epoch_')[-1])
    weights_lst = sorted(
        glob(os.path.join(args.ckpt_folder, '*.pth')) if args.ckpt_folder else [args.ckpt],
        key=_epoch_of,
        reverse=True
    )
    try:
        if args.resolution in [None, 'None', 0, '']:
            # Use original resolution for inference.
            data_size = None
        elif args.resolution in ['config.size']:
            data_size = config.size
        else:
            data_size = [int(l) for l in args.resolution.split('x')]
    except Exception as e:
        # e.__traceback__ can be None on bare exceptions (no raise frame).
        tb = e.__traceback__
        line = tb.tb_lineno if tb is not None else '?'
        print(f"Exception: {type(e).__name__} at line {line} of {__file__}: {e}")
        # default as the config.size.
        data_size = config.size

    for testset in args.testsets.split('+'):
        print('>>>> Testset: {}...'.format(testset))
        # pin_memory only helps CPU→GPU transfers. With CUDA disabled (or
        # when running on CPU) it costs page-locked memory for no benefit
        # and can warn loudly.
        _pin = torch.cuda.is_available() and str(config.device) != 'cpu'
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(testset, data_size=data_size, is_train=False),
            batch_size=config.batch_size_valid, shuffle=False, num_workers=config.num_workers, pin_memory=_pin
        )
        for weights in weights_lst:
            # The previous `% 1 != 0` filter was always False (dead code). The
            # original intent was filtering by step; restoring that requires a
            # `--step` flag, which we don't have here. So just iterate every
            # checkpoint, which is what was happening anyway.
            print('\tInferencing {}...'.format(weights))
            # map_location=device drops the CPU staging buffer entirely on
            # CUDA runs — saves a host-to-device copy per checkpoint.
            state_dict = torch.load(weights, map_location=device, weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            stems = [os.path.splitext(w)[0] for w in weights.split(os.sep)[-2:]]
            inference(
                model, data_loader_test=data_loader_test, pred_root=args.pred_root,
                method='--'.join(stems) + '-reso_{}'.format('x'.join([str(s) for s in data_size])),
                testset=testset, device=config.device
            )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default=sorted(glob(os.path.join('ckpts', '*')))[-1], type=str, help='model folder')
    parser.add_argument('--pred_root', default='e_preds', type=str, help='Output folder')
    parser.add_argument('--resolution', default='default', type=str, help='WeixHei')
    parser.add_argument('--testsets',
                        default=config.testsets.replace(',', '+'),
                        type=str,
                        help="Test all sets: DIS5K -> 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
