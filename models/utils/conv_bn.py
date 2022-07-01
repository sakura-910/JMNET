import torch.nn as nn
import math


class Conv_BN_ReLU2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        super(Conv_BN_ReLU2, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.bn(self.conv(x))
