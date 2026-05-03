import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import laplacian
from huggingface_hub import PyTorchModelHubMixin

from config import Config
from dataset import class_labels_TR_sorted
from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk, ResBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.modules.aspp import ASPP, ASPPDeformable


def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation='b c (hg h) (wg w) -> (b hg wg) c h w'):
    """Split an image into a grid of patches via einops rearrange.

    einops requires the spatial dims be divisible by grid_h × grid_w. When
    the caller derives grid sizes from an internal feature shape, an HR
    input (e.g. 1023×1024) can leave a non-zero remainder and crash the
    rearrange. Pad bottom-right with replicate to the next multiple; the
    downstream F.interpolate at the call site resamples to the correct
    feature shape, so the pad doesn't bleed into the model output.
    """
    if patch_ref is not None:
        grid_h = max(1, image.shape[-2] // patch_ref.shape[-2])
        grid_w = max(1, image.shape[-1] // patch_ref.shape[-1])
    pad_h = (-image.shape[-2]) % grid_h
    pad_w = (-image.shape[-1]) % grid_w
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    return patches

def patches2image(patches, grid_h=2, grid_w=2, patch_ref=None, transformation='(b hg wg) c h w -> b c (hg h) (wg w)'):
    if patch_ref is not None:
        grid_h, grid_w = patch_ref.shape[-2] // patches[0].shape[-2], patch_ref.shape[-1] // patches[0].shape[-1]
    image = rearrange(patches, transformation, hg=grid_h, wg=grid_w)
    return image

class BiRefNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet",
    repo_url="https://github.com/ZhengPeng7/BiRefNet",
    tags=['Image Segmentation', 'Background Removal', 'Mask Generation', 'Dichotomous Image Segmentation', 'Camouflaged Object Detection', 'Salient Object Detection']
):
    def __init__(self, bb_pretrained=True):
        super(BiRefNet, self).__init__()
        self.config = Config()
        # Match the upstream pretrained checkpoints (trained with align_corners=True)
        # by default. New training can flip this in config.py.
        self._ac = bool(getattr(self.config, 'align_corners', True))
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)

        channels = self.config.lateral_channels_in_collection

        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], len(class_labels_TR_sorted))
            )

        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(channels[0]+sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])

        self.decoder = Decoder(channels)

        if self.config.freeze_bb:
            # Freeze the backbone...
            for key, value in self.named_parameters():
                if 'bb.' in key and 'refiner.' not in key:
                    value.requires_grad = False

    def _run_backbone(self, x):
        """Run the backbone with the right call sequence for both conv-based
        (vgg/resnet, exposed as conv1..conv4 attributes) and feature-pyramid
        (Swin / PVT / DINO, callable directly) backbones."""
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
            return x1, x2, x3, x4
        return self.bb(x)

    def forward_enc(self, x):
        x1, x2, x3, x4 = self._run_backbone(x)
        if self.config.mul_scl_ipt:
            B, C, H, W = x.shape
            x_pyramid = F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=self._ac)
            # Same backbone code path for both 'cat' and 'add'; previously the
            # 'add' branch hard-coded `self.bb(x_pyramid)` and crashed for the
            # vgg/resnet backbones that expose conv1..conv4.
            x1_, x2_, x3_, x4_ = self._run_backbone(x_pyramid)
            if self.config.mul_scl_ipt == 'cat':
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=self._ac)], dim=1)
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=self._ac)], dim=1)
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=self._ac)], dim=1)
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=self._ac)], dim=1)
            elif self.config.mul_scl_ipt == 'add':
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=self._ac)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=self._ac)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=self._ac)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=self._ac)
        class_preds = self.cls_head(self.avgpool(x4).view(x4.shape[0], -1)) if self.training and self.config.auxiliary_classification else None
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=self._ac),
                        F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=self._ac),
                        F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=self._ac),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4), class_preds

    def forward_ori(self, x):
        ########## Encoder ##########
        (x1, x2, x3, x4), class_preds = self.forward_enc(x)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]
        if self.training and self.config.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, class_preds

    def forward(self, x):
        scaled_preds, class_preds = self.forward_ori(x)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        self._ac = bool(getattr(self.config, 'align_corners', True))
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = eval(self.config.lat_blk)

        self.bbs_without_pyramid = ['vit', 'dino']
        self.use_pyramid_neck = any(bb_without_pyramid in self.config.bb for bb_without_pyramid in self.bbs_without_pyramid)
        if self.use_pyramid_neck:
            self.manually_controlled_decoder_in_channels = [c * (1 + int(self.config.mul_scl_ipt == 'cat')) for c in (1536, 768, 384, 192)]     # Use the channels of swin_v1_l as default.
            self.pyramid_neck_x4 = LateralBlock(channels[0], self.manually_controlled_decoder_in_channels[0])
            self.pyramid_neck_x3 = LateralBlock(channels[1], self.manually_controlled_decoder_in_channels[1])
            self.pyramid_neck_x2 = LateralBlock(channels[2], self.manually_controlled_decoder_in_channels[2])
            self.pyramid_neck_x1 = LateralBlock(channels[3], self.manually_controlled_decoder_in_channels[3])

        # Channel arithmetic for the input-patch (dec_ipt) and decoder blocks.
        # ipt_blk_out_channels stays defined either way so downstream uses
        # don't depend on dec_ipt branch coverage.
        ipt_blk_out_channels = [0, 0, 0, 0]
        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            ipt_blk_in_channels = [2**i*3 for i in (10, 8, 6, 4, 0)] if self.split else [3] * 5
            ipt_blk_out_channels = [[N_dec_ipt, channels[i]//8][ipt_cha_opt] for i in range(4)]
            self.ipt_blk5 = DBlock(ipt_blk_in_channels[0], ipt_blk_out_channels[0], inter_channels=ic)
            self.ipt_blk4 = DBlock(ipt_blk_in_channels[1], ipt_blk_out_channels[0], inter_channels=ic)
            self.ipt_blk3 = DBlock(ipt_blk_in_channels[2], ipt_blk_out_channels[1], inter_channels=ic)
            self.ipt_blk2 = DBlock(ipt_blk_in_channels[3], ipt_blk_out_channels[2], inter_channels=ic)
            self.ipt_blk1 = DBlock(ipt_blk_in_channels[4], ipt_blk_out_channels[3], inter_channels=ic)
        else:
            self.split = None

        if self.use_pyramid_neck:
            bb_neck_out_channels = [c for c in self.manually_controlled_decoder_in_channels]
        else:
            bb_neck_out_channels = channels.copy()
        dec_blk_out_channels = [c for c in bb_neck_out_channels[1:]] + [bb_neck_out_channels[-1] // 2]
        if self.config.dec_ipt:
            dec_blk_in_channels = [bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)] for i in range(len(bb_neck_out_channels))]
        else:
            # Without dec_ipt the decoder blocks don't get extra ipt channels.
            dec_blk_in_channels = list(bb_neck_out_channels)

        self.decoder_block4 = DecoderBlock(dec_blk_in_channels[0], dec_blk_out_channels[0])
        self.decoder_block3 = DecoderBlock(dec_blk_in_channels[1], dec_blk_out_channels[1])
        self.decoder_block2 = DecoderBlock(dec_blk_in_channels[2], dec_blk_out_channels[2])
        self.decoder_block1 = DecoderBlock(dec_blk_in_channels[3], dec_blk_out_channels[3])
        self.conv_out1 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[3] + (ipt_blk_out_channels[3] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        # Backbone+PyramidNeck --> lateral block --> DecoderBlock
        self.lateral_block4 = LateralBlock(bb_neck_out_channels[1], dec_blk_out_channels[0])
        self.lateral_block3 = LateralBlock(bb_neck_out_channels[2], dec_blk_out_channels[1])
        self.lateral_block2 = LateralBlock(bb_neck_out_channels[3], dec_blk_out_channels[2])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                self.gdt_convs_4 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[0], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))

                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                
                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        size_x1_to_x4_template = [(x.shape[2] // (2 ** i), x.shape[3] // (2 ** i)) for i in (2, 3, 4, 5)]
        if self.use_pyramid_neck:
            x1 = F.interpolate(x1, size=size_x1_to_x4_template[0], mode='bilinear', align_corners=self._ac)
            x1 = self.pyramid_neck_x1(x1)

            x2 = F.interpolate(x2, size=size_x1_to_x4_template[1], mode='bilinear', align_corners=self._ac)
            x2 = self.pyramid_neck_x2(x2)

            x3 = F.interpolate(x3, size=size_x1_to_x4_template[2], mode='bilinear', align_corners=self._ac)
            x3 = self.pyramid_neck_x3(x3)

            x4 = F.interpolate(x4, size=size_x1_to_x4_template[3], mode='bilinear', align_corners=self._ac)
            x4 = self.pyramid_neck_x4(x4)
        outs = []

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=x4, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=self._ac))), 1)
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                # >> GT: gdt label needs m4 even when ms_supervision is disabled
                # (without it the multiplication below crashes with `None * Tensor`).
                m4_dia = m4 if m4 is not None else self.conv_ms_spvn_4(p4)
                gdt_label_main_4 = gdt_gt * F.interpolate(m4_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=self._ac)
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=self._ac)
        _p3 = _p4 + self.lateral_block4(x3)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=self._ac))), 1)
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3 if m3 is not None else self.conv_ms_spvn_3(p3)
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=self._ac)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=self._ac)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=self._ac))), 1)
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2 if m2 is not None else self.conv_ms_spvn_2(p2)
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=self._ac)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=self._ac)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=self._ac))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=self._ac)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=self._ac))), 1)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision and self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))
