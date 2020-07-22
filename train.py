import torch
import argparse
from torch.utils.data import DataLoader
from data.ImagenetDataset import get_imagenet_datasets
from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
from densenet import *
from msdnet.models.msdnet import MSDNet

import os
import time

DATA_PATH = "data/imagenet_images"
BATCH_SIZE = 2


def get_msd_net_model():
    grFact = '1-2-4-4'
    bnFact = '1-2-4-4'
    obj = argparse.Namespace()
    obj.nBlocks = 1
    obj.nChannels = 32
    obj.base = 4
    obj.stepmode = 'even'
    obj.step = 4
    obj.growthRate = 16
    obj.grFactor = list(map(int, grFact.split('-')))
    obj.prune = 'max'
    obj.bnFactor = list(map(int, bnFact.split('-')))
    obj.bottleneck = True
    obj.data = 'ImageNet'
    obj.nScales = len(obj.grFactor)
    obj.reduction = 0.5 # compression of densenet
    return MSDNet(obj).train()


if __name__ == "__main__":


    train_loader, test_loader, _ = get_dataloaders_alt(
        DATA_PATH, 
        data="ImageNet", 
        use_valid=False, 
        save='save/default-{}'.format(time.time()),
        batch_size=1, 
        workers=2, 
        splits=['train', 'test'])

    #model = ResNet.resnet50()
    #model = densenet121()

    model = get_msd_net_model()


    #model.eval()
    for i, (input, target) in enumerate(train_loader):
        with torch.no_grad():
            print(input.shape)
            print(target.shape)
            print(target)
            pred = model(input)
            print(pred[0].shape)
            print(pred)
        break