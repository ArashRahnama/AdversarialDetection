#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:14:01 2019

@author: Arash Rahnama
"""
import torch
import torch.nn as nn
###############################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3*3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, 
                     dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    # 3*3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     bias=False)
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in Basic Block')
        # both self.conv1 annd self.downsaple layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
            
        out = self.conv2(out)
        out = self.bn2(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)
                
        out += identity
        out = self.relu(out)
        return out
###############################################################################
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # both self.conv1 annd self.downsaple layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
       
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)      
        return out
###############################################################################
class ResNet(nn.Module):
    # ResNet18
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group        
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple inidicates if we should replace
            # the 2 by 2 strides with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None ' 
                             'or a 3-element tuple, got{}'.format(replace_stride_with_dilation))
            
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                       dilate=replace_stride_with_dilation[1])        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # zero initialize the last BN in each residual branch
        # so that the residual branch starts with zeros, and each residual 
        # block behaves like an identity. This improves the model by 0.2~0.3% 
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        # change the number of filters for 1*1 conv
        self.conv11_stage1 = nn.Conv2d(in_channels=64, out_channels=1, 
                                       kernel_size=(1,1))
        self.conv11_stage2 = nn.Conv2d(in_channels=128, out_channels=1, 
                                       kernel_size=(1,1))
        self.conv11_stage3 = nn.Conv2d(in_channels=256, out_channels=1, 
                                       kernel_size=(1,1))
        self.conv11_stage4 = nn.Conv2d(in_channels=512, out_channels=1, 
                                       kernel_size=(1,1))

        # for fully connected detection layer
        self.detect_relu = nn.ReLU(inplace=False)
        detect_in_size = 4165 # 56*56+28*28+14*14+7*7
        self.detect_fc1 = nn.Linear(detect_in_size, 2000)
        self.detect_fc2 = nn.Linear(2000, 1000)
        self.detect_fc3 = nn.Linear(1000, 2)
        
        # if False, a regular ResNet, if True, a detection network
        self.defense_mode = False
        
    def defense_mode_on(self):
        self.defense_mode = True
    def defense_mode_off(self):
        self.defense_mode = False
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, 
                                               planes*block.expansion,
                                               stride),
                                        norm_layer(planes*block.expansion),)
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            self.groups, self.base_width, 
                            previous_dilation, norm_layer))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, 
                                dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
            
        if self.defense_mode:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
                
            x1 = self.conv11_stage1(x1)
            x2 = self.conv11_stage2(x2)
            x3 = self.conv11_stage3(x3)
            x4 = self.conv11_stage4(x4)
                
            x1 = x1.view(-1, 56*56)
            x2 = x2.view(-1, 28*28)
            x3 = x3.view(-1, 14*14)
            x4 = x4.view(-1, 7*7)
                
            x = torch.cat((x1, x2, x3, x4), 1)
                
            x = self.detect_relu(self.detect_fc1(x))
            x = self.detect_relu(self.detect_fc2(x))
            x = self.detect_fc3(x)
            return x
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
                
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x
###############################################################################
def _ResNet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model
def ResNet18(**kwargs):
    # construct a ResNet18 Model
    return _ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    # construct a ResNet34 Model
    return _ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
