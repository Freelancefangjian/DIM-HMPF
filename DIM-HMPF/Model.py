import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
import math



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)






class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
class multilayer(nn.Module):
    def __init__(self, dim):
        super(multilayer, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=(0, 0))
        self.Conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=(1, 1))
        self.Conv5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=(2, 2))
        self.Conv = nn.Conv2d(in_channels=3*dim, out_channels=dim, kernel_size=1, padding=(0, 0))
    def forward(self, x):
        x1 = self.Conv1(x)
        x3 = self.Conv3(x)
        x5 = self.Conv5(x)
        x = torch.cat([x1, x3, x5], dim=1)
        x = self.Conv(x)
        return x
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class detail_fusion(nn.Module):
    def __init__(self, out_channel):
        super(detail_fusion, self).__init__()
        self.Conv62_31 = nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=3, padding=(1, 1))
        self.Conv31_31 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=3, padding=(1, 1))
        self.re = nn.ReLU(inplace=True).cuda()
    def forward(self, x, y):
        x = self.Conv31_31(x)
        x = self.re(x)
        x = self.Conv31_31(x)

        y = self.Conv31_31(y)
        y = self.re(y)
        y = self.Conv31_31(y)

        z = self.Conv62_31(torch.cat((x, y), axis=1))
        z = self.re(z)
        z = self.Conv31_31(z)
        return z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.upSample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upSample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Conv32_31 = nn.Conv2d(in_channels=32, out_channels=31, kernel_size=3, padding=(1, 1))
        self.Conv31_31 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1))
        self.Conv3_31 = nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, padding=(1, 1))
        self.eta = torch.nn.Parameter(torch.tensor(0.1))
        self.mu = torch.nn.Parameter(torch.tensor(0.1))
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))
        self.beta = torch.nn.Parameter(torch.tensor(0.1))
        self.n = 5
        self.detail = detail_fusion(out_channel = 128)
        self.re = nn.ReLU(inplace=True).cuda()
    def forward(self, HSI, MSI, PAN):
        Up_HSI = self.upSample(HSI)
        PAN1 = PAN
        for i in range(127):
            PAN1 = torch.cat((PAN1,PAN),axis=1)
        D_lamda_0 = PAN1-Up_HSI
        MSI1 = self.Conv3_31(MSI)
        MSI1 = self.upSample1(MSI1)
        D_v_0 = MSI1-Up_HSI
        S_0 = (D_lamda_0 + D_v_0)/2
        P_D = D_lamda_0
        M_D = D_v_0
        D = S_0
        X = Up_HSI
        DD = self.detail(P_D,M_D)
        for i in range(self.n):
            P_D = P_D - self.eta*(P_D-(PAN1-self.Conv31_31(Up_HSI))+self.mu*torch.mul(self.Conv31_31(P_D),(DD - D)))
            DD = self.detail(P_D,M_D)
            M_D = M_D - self.eta*(self.alpha*(M_D-(MSI1-self.Conv31_31(Up_HSI)))+self.mu*torch.mul(self.Conv31_31(M_D),(DD - D)))
            DD = self.detail(P_D,M_D)
            D = D - self.eta* (self.beta*(D+Up_HSI-X)+self.mu*(DD - D))
            X = X - self.eta*self.beta*(D+Up_HSI-X)
        X = Up_HSI + X + D
        return X
