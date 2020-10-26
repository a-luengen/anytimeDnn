import unittest
import os
from .context import msdnet
from .context import data_utils
from .context import utils

class ArgsObj(object):
    pass

class TestMsdNetCreation(unittest.TestCase):
    @classmethod
    def createArgsObject(self, args_dict):
        for key in args_dict:
            setattr(self.args, key, args_dict[key])

    @classmethod
    def setUpClass(self):
        self.args = ArgsObj()

        growFactor = list(map(int, "1-2-4-4".split("-")))
        bnFactor = list(map(int, "1-2-4-4".split("-")))

        args_dict = {
            'gpu': 'gpu:0',
            'use_valid': True,
            'data': 'ImageNet',
            'save': 'save/',
            'evalmode': None,
            'epoch': 0,
            'epochs': 90,
            'arch': 'msdnet',
            'seed': 42,
            'test_interval': 10,

            'grFactor': growFactor,
            'bnFactor': bnFactor,
            'nBlocks': 5,
            'nChannels': 32,
            'nScales': len(growFactor),
            'reduction': 0.5,
            'bottleneck': True,
            'prune': 'max',
            'growthRate': 16,
            'base': 4,
            'step': 4,
            'stepmode': 'even',
            
            'lr': 0.1,
            'lr_type': 'multistep',
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'resume': False,
            'data_root': 'data/imagenet_red/',
            'batch_size': 4,
            'workers': 1,
            'print_freq': 2
        } 

        self.createArgsObject(args_dict)
        
    def test000_createMsdNetDirectly_noExcpetion(self):
        msdnet.models.msdnet(self.args)

    
    def test010_createMsdNet_WithFourBlocks_FromUtils_GetModelsWithOptimized_ReturnsCorrectModel(self):
        model = utils.getModelWithOptimized('msdnet4')
        expected_blocks = 4
        expected_classifiers = 4
        self.assertEqual(expected_blocks, len(model.blocks))
        self.assertEqual(expected_classifiers, len(model.classifier))

    def test020_createMsdNet_WithFiveBlocks_FormUtils_getModelsWithOptimized_ReturnsCorrectModel(self):
        model = utils.getModelWithOptimized('msdnet')
        expected_blocks = 5
        expected_classifiers = 5
        self.assertEqual(expected_blocks, len(model.blocks))
        self.assertEqual(expected_classifiers, len(model.classifier))