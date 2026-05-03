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

        # torchvision.ops.deform_conv2d does not implement a bf16 CUDA kernel;
        # it raises NotImplementedError under autocast(bfloat16). Cast to fp32
        # for the op and back so the module is usable inside bf16 autocast.
        in_dtype = x.dtype
        if in_dtype == torch.bfloat16:
            x_f = x.float()
            offset_f = offset.float()
            modulator_f = modulator.float()
            weight_f = self.regular_conv.weight.float()
            bias_f = self.regular_conv.bias.float() if self.regular_conv.bias is not None else None
            out = deform_conv2d(
                input=x_f, offset=offset_f, weight=weight_f, bias=bias_f,
                padding=self.padding, mask=modulator_f, stride=self.stride,
            )
            return out.to(in_dtype)

        return deform_conv2d(
            input=x, offset=offset,
            weight=self.regular_conv.weight, bias=self.regular_conv.bias,
            padding=self.padding, mask=modulator, stride=self.stride,
        )
