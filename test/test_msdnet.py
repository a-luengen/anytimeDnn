import unittest
import torch
import os
import shutil

from .context import msdnet
from .context import utils
from .context import train_acc
from .context import data_loader

class Object(object):
    pass

class TestMSDNet(unittest.TestCase):

    imagenet_args = Object()

    test_checkpoint_dir = "test/checkpoint"

    def setUp(self):
        args = Object()
        test_growFact = "1-2-4-4".split("-")
        test_bnFact = "1-2-4-4".split("-")
        args_dict = {
            'grFactor': list(map(int, test_growFact)),
            'bnFactor': list(map(int, test_bnFact)),
            'nBlocks': 5,
            'nChannels': 32,
            'base': 4,
            'stepmode': 'even',
            'step': 4,
            'growthRate': 16,
            'prune': 'max',
            'bottleneck': True,
            'data': 'ImageNet',
            'nScales': len(test_growFact),
            'reduction': 0.5
        }
        for key in args_dict:
            setattr(args, key, args_dict[key])

        self.imagenet_args = args

    def tearDown(self):
        if os.path.isdir(self.test_checkpoint_dir):
            shutil.rmtree(self.test_checkpoint_dir)


    def test000_getMSDNetFromUtils_noException(self):
        net = utils.get_msd_net_model()

    def test010_createMSDNetForImageNet_noException(self):

        self.assertTrue(len(self.imagenet_args.grFactor) == 4)
        self.assertTrue(len(self.imagenet_args.bnFactor) == 4)

        net = msdnet.models.msdnet(self.imagenet_args)
        
        self.assertIsNotNone(net)
    
    def test020_forwardingOfMSDNet_forImagenet_noException(self):
        net = msdnet.models.msdnet(self.imagenet_args)

        test_img = torch.rand(1, 3, 224, 224)

        res = net(test_img)

        self.assertIsNotNone(res)
        self.assertEqual(len(res), self.imagenet_args.nBlocks)


    def test030_calculateAccuracyOfMsdNetOutput_noException(self):
        net = msdnet.models.msdnet(self.imagenet_args)

        loader, _, _ = data_loader.get_zipped_dataloaders(data_loader.REDUCED_SET_PATH, 1)

        (test_img, test_target) = next(iter(loader))

        output = net(test_img)

        if not isinstance(output, list):
            output = [output]

        for i in range(len(output)):
            train_acc(output[i].data, test_target, topk=(1,5))

    def test040_CreateAndResumeCheckpoint_NoException(self):
        net = msdnet.models.msdnet(self.imagenet_args)

        test_epoch = 4
        test_arch = 'msdnet'

        test_best_acc = 4.2

        test_lr = 0.001
        test_mom = 0.9
        test_weight_decay = 0.1

        test_optim = torch.optim.SGD(net.parameters(), lr=test_lr, momentum=test_mom, weight_decay=test_weight_decay)

        stateDict = utils.getStateDict(net, test_epoch, test_arch, test_best_acc, test_optim)

        utils.save_checkpoint(stateDict, False, test_arch, self.test_checkpoint_dir)

        net = utils.resumeFromPath(self.test_checkpoint_dir, 'msdnet', test_optim)

        