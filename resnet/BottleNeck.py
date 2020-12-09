import torch.nn as nn
#from resnet.OCL_Convolution import OCL_Convolution

import logging

class CustIdent(nn.Identity):
    def forward(self, x):
        logging.info("Forward was called")
        return x

class BottleNeck(nn.Module):
    """
        A Residual block for resnet over 50 layers
    """
    dropPolicy = None
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_ocl=False, dropResidualPolicy=None, layer_nr=None):
        super().__init__()
        
        if dropResidualPolicy is not None: 
            self.dropPolicy = dropResidualPolicy
            self.layer_nr = layer_nr

        # defining pipeline for residual calculation
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            #OCL_Convolution(in_channels, out_channels, kernel_size=1, bias=False, use_ocl=use_ocl),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        # adjust shortcut values to match with residual
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        if self.dropPolicy and self.dropPolicy.shouldDrop(self.layer_nr):
        #if self.dropPolicy and self.shouldDrop:
            return nn.ReLU(inplace=True)(self.shortcut(x))
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    