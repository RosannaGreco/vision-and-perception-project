#this file defines a Residual Network model 

import torch
from torch import nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample= None, stride = 1):
        super(ResidualBlock,self).__init__()

        self.expansion = 4
        self.downsample = downsample


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding = 0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)




    def forward(self, input):
        shortcut = input
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input =(self.bn3(self.conv3(input)))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        input = input + shortcut
        input= nn.ReLU()(input)
        return input










class ResNet(nn.Module):
    def __init__(self, layers, in_channels, outputs):
        super(ResNet,self).__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 7, stride= 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride =2, padding = 1)

    # residual blocks-------------------------------------------------------------
        self.layer1 = self._reslayer_maker(layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._reslayer_maker(layers[1], out_channels=128, stride=2)
        self.layer3 = self._reslayer_maker(layers[2], out_channels=256, stride=2)
        self.layer4 = self._reslayer_maker(layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*4, 120)
        self.fc2 = nn.Linear(120, outputs)

       


    def forward(self, input):
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = self.maxpool(input)

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.avgpool(input)
        input = input.reshape(input.shape[0], -1)
        
        input = self.fc1(input)
        
        input = self.fc2(input)
        return input

    def _reslayer_maker(self, num_res_blocks, out_channels, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_ch != out_channels*4:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, out_channels*4, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(out_channels*4))

        layers.append(ResidualBlock(self.in_ch, out_channels, downsample, stride))
        self.in_ch = out_channels*4

        for i in range(num_res_blocks -1):
            layers.append(ResidualBlock(self.in_ch, out_channels))
        return nn.Sequential(*layers)






def ResNet152(in_channels = 3, outputs = 2):
    return ResNet([3,8,36,3], in_channels, outputs)







