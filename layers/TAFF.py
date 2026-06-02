import torch
import torch.nn as nn
import torch.nn.functional as F

#Dual-Space Task-Adaptive Features Fusion
def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class TAFF(nn.Module):
    def __init__(self, dim, vis=False):
        super(TAFF, self).__init__()
        self.dim = dim

        compress_c = dim//2

        self.weight_0 = add_conv(self.dim, compress_c, 1, 1)
        self.weight_1 = add_conv(self.dim, compress_c, 1, 1)
        self.weights = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.vis = vis

    def forward(self, x_0, x_1):
        weight_v_0 = self.weight_0(x_0)
        weight_v_1 = self.weight_1(x_1)
        weight_v = torch.cat((weight_v_0, weight_v_1), 1)
        weight = self.weights(weight_v)
        weight = F.softmax(weight, dim=1)

        fused_out = x_0 * weight[:, 0:1, :, :] + x_1 * weight[:, 1:2, :, :]

        if self.vis:
            return fused_out, weight
        else:
            return fused_out