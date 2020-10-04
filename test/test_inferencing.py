import unittest 
import torch
import torch.nn as nn
from utils import getModel
from msdnet.dataloader import get_dataloaders_alt
from data.ImagenetDataset import get_zipped_dataloaders
import os
from data.utils import getClassToIndexMapping

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
            test_criterion(output, target)
            if i == 0: break

    def test020_testDenseNet169Output_withLoss_noException(self):
        test_batch = 1
        test_loader, _, _ = get_zipped_dataloaders(self.TEST_DATASET_PATH, test_batch)
        
        test_criterion = nn.CrossEntropyLoss()

        model = getModel('densenet169')

        for i, (img, target) in enumerate(test_loader):
            output = model(img)
            test_criterion(output, target)
            if i == 0: break

    def test030_labelAndIndexMapping(self):
        test_batch = 1
        test_loader, _, _ = get_zipped_dataloaders(self.TEST_DATASET_PATH, test_batch)
        img, target = next(iter(test_loader))
        
        index_path = os.path.join(self.TEST_DATASET_PATH, 'index-train.txt')
        class_to_global_index = getClassToIndexMapping(index_path)

        label_to_class = list(set(class_to_global_index))
        label_to_class.sort()

        self.assertEqual(len(label_to_class), 40)
        self.assertEqual(len(class_to_global_index), len(test_loader))
    
        index_path = os.path.join(self.TEST_DATASET_PATH, 'index-val.txt')
        class_to_global_val_index = getClassToIndexMapping(index_path)

        label_to_class_val = list(set(class_to_global_val_index))
        label_to_class_val.sort()
        self.assertEqual(len(label_to_class_val), len(label_to_class))
        self.assertEqual(label_to_class_val, label_to_class)

