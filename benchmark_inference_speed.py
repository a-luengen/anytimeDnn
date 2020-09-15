import os
import shutil
import time
from timeit import default_timer as timer
import datetime
import sys
import logging

from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
import densenet.densenet as dn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from utils import *


def runBenchForModel(arch: str):
    logging.info(f'Running Benchmark for {arch} network.')

    start_init = timer()
    model = getModel(arch)
    logging.info(f'Time to init {arch} network: {timer() - start_init:.4f} seconds')

    model.eval()

    start_inference = timer()
    output = model(imagenet_test)
    logging.info(f'Time for inferencing {arch} network: {timer() - start_inference:.4f} seconds')

    if arch == 'msdnet':
        logging.info(f'Size of ')

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    
    #architectures = ['resnet50', 'resnet101', 'resnet152', 'densenet121', 'msdnet']
    architectures = ['resnet50', 'resnet50-pol']
    imagenet_test = torch.rand((1, 3, 224 ,224))

    # run benchmark loop
    for i, arch in enumerate(architectures):
        runBenchForModel(arch)