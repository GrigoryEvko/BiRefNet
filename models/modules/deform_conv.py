import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        # torchvision.ops.deform_conv2d has its own autocast wrapper that
        # silently promotes everything to fp32 — under bf16 autocast the
        # entry-point dispatcher gives back fp32 even when our inputs were
        # bf16. That fp32 output then re-promotes the downstream `cat` and
        # implicit casts ripple through the decoder.
        #
        # Gating the cast on `x.dtype == bf16` is wrong: x stays fp32 inside
        # autocast (autocast doesn't touch the input), while offset / modulator
        # come from Conv2d outputs that ARE autocast-promoted to bf16. So the
        # tensors entering the call are mixed-dtype and the previous gate
        # never fired in production.
        #
        # Capture the autocast-effective dtype from `offset` (a conv output —
        # reflects what autocast wants), disable autocast around the deform
        # call so torchvision's wrapper doesn't fight us, run in fp32, then
        # restore the target dtype.
        target_dtype = offset.dtype
        device_type = x.device.type if x.is_cuda or x.device.type == "cpu" else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_f = x.float()
            offset_f = offset.float()
            modulator_f = modulator.float()
            weight_f = self.regular_conv.weight.float()
            bias_f = self.regular_conv.bias.float() if self.regular_conv.bias is not None else None
            out = deform_conv2d(
                input=x_f, offset=offset_f, weight=weight_f, bias=bias_f,
                padding=self.padding, mask=modulator_f, stride=self.stride,
            )
        return out.to(target_dtype) if out.dtype != target_dtype else out
