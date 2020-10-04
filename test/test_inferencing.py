import unittest 
import torch
import torch.nn as nn
from utils import getModel
from msdnet.dataloader import get_dataloaders_alt
from data.ImagenetDataset import get_zipped_dataloaders
import os

class TestInferencing(unittest.TestCase):

    TEST_DATASET_PATH = os.path.join(os.getcwd(), "data", "imagenet_red")
    TEST_DATASET_PATH_ALT = "data/imagenet_images"

    def test000_testDenseNet121Output_withLoss_noException(self):
        test_batch = 1
        test_loader, _, _ = get_zipped_dataloaders(self.TEST_DATASET_PATH, test_batch)
        
        test_criterion = nn.CrossEntropyLoss()
        

        model = getModel('densenet121')
        for i, (img, target) in enumerate(test_loader):
            output = model(img)
            loss = test_criterion(output, target)
            if i == 0: break

    def test020_testDenseNet169Output_withLoss_noException(self):
        test_batch = 1
        test_loader, _, _ = get_zipped_dataloaders(self.TEST_DATASET_PATH, test_batch)
        
        test_criterion = nn.CrossEntropyLoss()

        model = getModel('densenet169')

        for i, (img, target) in enumerate(test_loader):
            output = model(img)
            loss = test_criterion(output, target)
            if i == 0: break

