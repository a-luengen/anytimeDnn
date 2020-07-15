import torch
from torch.utils.data import DataLoader
from data.ImagenetDataset import get_imagenet_datasets
from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
from densenet import *
#from msdnet.models.msdnet import MSDNet

import os
import time

DATA_PATH = "data/imagenet_images"
BATCH_SIZE = 2


if __name__ == "__main__":


    train_loader, test_loader, _ = get_dataloaders_alt(
        DATA_PATH, 
        data="ImageNet", 
        use_valid=False, 
        save='save/default-{}'.format(time.time()),
        batch_size=1, 
        workers=2, 
        splits=['train', 'test'])

    # densenet - train cifar:
    #  for i, (input, target) in enumerate(val_loader):

    # msdnet:
    # for i, (input, target) in enumerate(train_loader):

    print(len(train_loader))
    print(len(test_loader))
    #model = ResNet.resnet34()
    model = densenet121()
    #model.eval()
    for i, (input, target) in enumerate(train_loader):
        print(input.shape)
        print(target.shape)
        print(target)
        pred = model(input)
        print(pred[0].shape)
        print(pred)
        break