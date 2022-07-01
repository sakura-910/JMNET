import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU
from .eca_layer import eca_layer

class SS_ASPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SS_ASPM, self).__init__()

        #self.in_channels = in_channels

        self.reduce_layer1 = Conv_BN_ReLU(64, 128)
        self.reduce_layer2 = Conv_BN_ReLU(128, 128)
        self.reduce_layer3 = Conv_BN_ReLU(256, 128)
        self.reduce_layer4 = Conv_BN_ReLU(512, 128)
#lfa/eca+lfa
        self.out_channels = out_channels

        self.dwconv3_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(self.out_channels, self.out_channels)

        self.dwconv2_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(self.out_channels, self.out_channels)

        self.dwconv1_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(self.out_channels, self.out_channels)

        self.dwconv2_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(self.out_channels, self.out_channels)

        self.dwconv3_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(self.out_channels, self.out_channels)

        self.dwconv4_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, groups=self.out_channels, bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(self.out_channels, self.out_channels)
#eca
        self.eca_layer1 = eca_layer(64, 3)  #3
        self.eca_layer2 = eca_layer(128, 5)   #5
        self.eca_layer3 = eca_layer(256, 5)    #5
        self.eca_layer4 = eca_layer(512, 5)    #5

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, f1, f2, f3, f4):
#eca/eca+lfa
        f11 = self.eca_layer1(f1)
        f21 = self.eca_layer2(f2)
        f31 = self.eca_layer3(f3)
        f41 = self.eca_layer4(f4)

        f11_1 = self.reduce_layer1(f11)
        # print(f1)
        f21_1 = self.reduce_layer2(f21)
        f31_1 = self.reduce_layer3(f31)
        f41_1 = self.reduce_layer4(f41)
#lfa/eca+lfa
        f1 = self.reduce_layer1(f1)
        #print(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)

        f3_1 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_1 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_1, f2)))
        f1_1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_1, f1)))

        f2_1 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_1, f1)))
        f3_1 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_1, f2_1)))
        f4_1 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_1)))
#lfa
#         f1 = f1_1 + f1
#         f2 = f2_1 + f2
#         f3 = f3_1 + f3
#         f4 = f4_1 + f4
#eca+lfa
        f1_1 = f1_1 + f1
        f2_1 = f2_1 + f2
        f3_1 = f3_1 + f3
        f4_1 = f4_1 + f4

        f1 = f11_1 + f1_1
        f2 = f21_1 + f2_1
        f3 = f31_1 + f3_1
        f4 = f41_1 + f4_1
#
# #lfa/eca+lfa
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)  #512

#eca
        # f2 = self._upsample(f21_1, f11_1.size())
        # f3 = self._upsample(f31_1, f11_1.size())
        # f4 = self._upsample(f41_1, f11_1.size())
        # f = torch.cat((f11_1, f2, f3, f4), 1)  #512


        return f
