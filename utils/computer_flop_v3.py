import torch.nn.functional as F
from torchstat import stat
import os
import sys
import torch
import torch.nn as nn
import math
from models.backbone.eca_module import eca_layer
from models.utils import Conv_BN_ReLU
from models.neck.eca_layer import eca_layer
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] ='1'

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['eca_resnet18']


model_urls = {
    'eca_resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,stride=1,downsample=None, k_size=3):
        super(ECABasicBlock,self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        #self.esa = esa_layer(planes)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        #out = self.esa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes*4, k_size)
        #self.esa = esa_layer(planes*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)
        #out = self.esa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 5, 5, 5]):

        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
                               #bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)      #kuowei
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)

#neck
        self.reduce_layer1 = Conv_BN_ReLU(64, 128)
        self.reduce_layer2 = Conv_BN_ReLU(128, 128)
        self.reduce_layer3 = Conv_BN_ReLU(256, 128)
        self.reduce_layer4 = Conv_BN_ReLU(512, 128)

        self.out_channels = 128

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


#pa_head
        self.convv1 = nn.Conv2d(512, 128,kernel_size=3, stride=1, padding=1)
        self.bnv1 = nn.BatchNorm2d(128)
        self.reluv1 = nn.ReLU(inplace=True)

        self.convv2 = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')


    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        f = []
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)

        f = tuple(f)


        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        f3_1 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_1 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_1, f2)))
        f1_1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_1, f1)))

        f2_1 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_1, f1)))
        f3_1 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_1, f2_1)))
        f4_1 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_1)))

        f1 = f1_1 + f1
        f2 = f2_1 + f2
        f3 = f3_1 + f3
        f4 = f4_1 + f4

        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())

        f = torch.cat((f1, f2, f3, f4), 1)  #512


        out = self.convv1(f)
        out = self.reluv1(self.bnv1(out))
        out = self.convv2(out)
        #
        return out




def eca_resnet18(k_size=[3, 5, 5, 5], pretrained=False, **kwargs):
    """Constructs a Resnet-18 model.

    Args:
        k_size:Addptive selection of kernel size
        pretrained(bool):If True, returns a model pre-trained on ImageNet
        #pretrained (bool): If True, returns a model pre-trained on Places
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], k_size=k_size, **kwargs)
    #model.avgpool = nn.AdaptiveMaxPool2d(1)
    #model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['eca_resnet18']), strict=False)
    return model

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)






if __name__ == '__main__':
    model = eca_resnet18()
    stat(model, (3, 768, 768))