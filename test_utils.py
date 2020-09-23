import unittest
import os
import shutil
import random
import torchvision.models as models
from utils import *

class TestUtilFunctions(unittest.TestCase):

    test_dir = os.path.join(os.getcwd(), 'test_checkpoints')

    arch_name = 'resnet18'

    test_net = None

    test_optim = None

    def setUp(self):
        self.test_net = models.resnet18()
        self.test_optim = torch.optim.SGD(
            self.test_net.parameters(), 
            0.1, 
            momentum=0.9, 
            weight_decay=10)

    def tearDown(self):
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test01_testCheckpointFunction_noException(self):

        state_dict = getStateDict(self.test_net, 0, self.arch_name, 0.0, self.test_optim)

        save_checkpoint(state_dict, False, self.arch_name, self.test_dir)

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, self.arch_name + CHECKPOINT_POSTFIX)))
    
    def test01_testCheckpointFunction_withIsBestTrue_noException(self):

        state_dict = getStateDict(self.test_net, 0, self.arch_name, 0.0, self.test_optim)

        save_checkpoint(state_dict, True, self.arch_name, self.test_dir)

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, self.arch_name + CHECKPOINT_BEST_POSTFIX)))
    
    def test02_getStateDict_CorrectParametersSet(self):
        test_acc = random.random()
        test_epoch = random.randint(0, 100)

        state_dict = getStateDict(self.test_net, test_epoch, self.arch_name, test_acc, self.test_optim)

        self.assertEqual(state_dict['epoch'], test_epoch)
        self.assertEqual(state_dict['arch'], self.arch_name)
        self.assertEqual(state_dict['best_acc'], test_acc)
        self.assertIsNotNone(state_dict['optimizer'])
        self.assertIsNotNone(state_dict['state_dict'])

    def test03_testResumeCheckpointFunction_returnsCorrectParameters(self):
        is_best = False
        test_acc = 1.2
        test_epoch = 5

        state_dict = getStateDict(self.test_net, test_epoch, self.arch_name, test_acc, self.test_optim)

        save_checkpoint(state_dict, is_best, self.arch_name, self.test_dir)

        self.test_net, self.optim, epoch, best_prec = resumeFromPath(
            os.path.join(self.test_dir, self.arch_name + CHECKPOINT_POSTFIX), 
            self.test_net,
            self.test_optim)

        self.assertEqual(epoch, test_epoch)
        self.assertEqual(best_prec, test_acc)
    
    def test04_testResumeCheckpointFunction_WithBest_returnsCorrectParameters(self):
        is_best = True
        test_acc = 1.2
        test_epoch = 5

        state_dict = getStateDict(self.test_net, test_epoch, self.arch_name, test_acc, self.test_optim)

        save_checkpoint(state_dict, is_best, self.arch_name, self.test_dir)

        self.test_net, self.test_optim, epoch, best_prec = resumeFromPath(
            os.path.join(self.test_dir, self.arch_name + CHECKPOINT_BEST_POSTFIX), 
            self.test_net,
            self.test_optim)

        self.assertEqual(epoch, test_epoch)
        self.assertEqual(best_prec, test_acc)
    
    def test05_testResumeCheckpointFunction_returnsCorrectParameter_onNoCheckpointFound(self):

        result_net, result_optim, epoch, best_prec = resumeFromPath("resnet_18.pth.tar", self.test_net, self.test_optim)

        self.assertEqual(result_net, self.test_net)
        self.assertEqual(result_optim, self.test_optim)
        self.assertEqual(epoch, 0)
        self.assertEqual(best_prec, 0.0)

    def test06_architectureParameterAmount_SameAsPytorchImpl(self):  
        testDenseNet = getModel('densenet121')
        test_count = sum([p.data.nelement() for p in testDenseNet.parameters()])
        torchDenseNet = torchvision.models.densenet121(num_classes=40)
        torch_count = sum([p.data.nelement() for p in torchDenseNet.parameters()])
        self.assertEqual(test_count, torch_count)
