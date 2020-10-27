import torch
import torch.nn as nn
from resnet.BottleNeck import BottleNeck
from resnet.BasicBlock import BasicBlock
#from resnet.OCL_Convolution import OCL_Convolution
import logging
from resnet.DropPolicies import *

class ResNet(nn.Module):
    """
        Modified ResNet class to use the Drop Policies to test Layer-Skipping techniques.
    """

    def __init__(self, block, num_block, num_classes=40, use_policy=False):
        super().__init__()
        self.in_channels = 64
        self.skipable_layer_count = 0

        if use_policy: 
            self.dropPolicy = getDropPolicy()
        else:
            self.dropPolicy = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            #OCL_Convolution(3, 64, kernel_size=3, padding=1, bias=False, use_ocl=use_ocl),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.dropPolicy is not None:
            self.dropPolicy.setMaxSkipableLayers(self.skipable_layer_count)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron network layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            #layers.append(block(self.in_channels, out_channels, stride, use_ocl=self.use_ocl))
            layers.append(block(self.in_channels, out_channels, stride, dropResidualPolicy=self.dropPolicy))
            self.in_channels = out_channels * block.expansion
            self.skipable_layer_count += 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.dropPolicy:
            self.dropPolicy.reset()
        return output 

# Functions to generate a model with the needed amount of trainable parameters

def resnet18(use_policy=False):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], use_policy=use_policy)

def resnet34(use_policy=False):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], use_policy=use_policy)

def resnet50(use_policy=False):
    """ return a ResNet 50 object with opencl supported inferencing
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], use_policy=use_policy)

def resnet101(use_policy=False):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], use_policy=use_policy)

def resnet152(use_policy=False):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], use_policy=use_policy)