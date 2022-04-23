import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=4, aux_logits=False, init_weights=False):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.conv1(x)  # [None, 3, 224, 224] -> [None, 64, 112, 112]
        x = self.pool1(x)  # [None, 64, 112, 112] -> [None, 64, 56, 56]
        x = self.conv2(x)
        x = self.conv3(x)  # [None, 64, 112, 112] -> [None, 192, 56, 56]
        x = self.pool2(x)  # [None, 192, 56, 56] -> [None, 192, 28, 28]

        x = self.inception3a(x) # [None, 192, 28, 28] -> [None, 256, 28, 28]
        x = self.inception3b(x)  # [None, 256, 28, 28] -> [None, 480, 28, 28]
        x = self.pool3(x)  # [None, 480, 28, 28] -> [None, 480, 14, 14]
        x = self.inception4a(x) # [None, 480, 14, 14] -> [None, 512, 14, 14]
        if self.training and self.aux_logits:  # eval mode discards this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x) # [None, 512, 14, 14] -> [None, 528, 14, 14]
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)  # [None, 528, 14, 14] -> [None, 832, 14, 14]
        x = self.pool4(x) # [None, 832, 14, 14] -> [None, 832, 7, 7]
        x = self.inception5a(x)
        x = self.inception5b(x)  # [None, 832, 7, 7] -> [None, 1024, 7, 7]

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output size [batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.avgpool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # input 224 224
    model = GoogLeNet();
    x = torch.Tensor(np.zeros((128, 1, 224, 224)));
    print(model(x).shape)
    

