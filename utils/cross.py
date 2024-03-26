import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.png")
    Image.fromarray(np.array(result)).save(im_save_path)
    

class DPM(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(DPM, self).__init__()  
        self.conv_head = nn.Conv2d(in_channels, out_channels,1)
 
        self.conv3_1_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.cca =CCALayer(out_channels)
        self.depthwise = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels)
        self.depthwise_dilated = nn.Conv2d(out_channels, out_channels, 5,stride=1,padding=6, groups=out_channels,dilation=3)
        self.conv_tail = nn.Conv2d(out_channels,in_channels,1)
        self.active = nn.Sigmoid()
    def forward(self, input):
        # print(input.shape)
        input_h = self.conv_head(input)
        x = self.conv3_1_A(input_h) + self.conv3_1_B(input_h) 

        x_cca = self.cca(x)
        x_de = self.depthwise(x_cca+input_h)
        x_de = self.depthwise_dilated(x_de)
        x_fea = x_de + x
        x_fea = self.active(self.conv_tail(x_de))
        return (x_fea *input)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

def stdv_channels(F):
    assert (F.dim() == 3)
    F_mean = mean_channels(F)
    eps = 1e-7
    F_variance = (F - F_mean+eps).pow(2).sum(2, keepdim=True).sum(1, keepdim=True) / (F.size(1) * F.size(2))
    return F_variance.pow(0.5)
def mean_channels(F):
    assert(F.dim() == 3)
    spatial_sum = F.sum(2, keepdim=True).sum(1, keepdim=True)
    return spatial_sum / (F.size(1) * F.size(2))

