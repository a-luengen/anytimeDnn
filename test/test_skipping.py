import unittest
import torch

from .context import resnet
from .context import resnet_utils

class TestSkippingPolicies(unittest.TestCase):

    def test00_resnet50_noExceptionOnCreation(self):
        test_net = resnet.ResNet.resnet50()
        self.assertIsNotNone(test_net)
        test_net = resnet.ResNet.resnet50(use_policy=True)
        self.assertIsNotNone(test_net)
    
    def test01_resnet101_noExceptionOnCreation(self):
        test_net = resnet.ResNet.resnet101()
        self.assertIsNotNone(test_net)
        test_net = resnet.ResNet.resnet101(use_policy=True)
        self.assertIsNotNone(test_net)
    
    def test02_resnet152_noExceptionOnCreation(self):
        test_net = resnet.ResNet.resnet152()
        self.assertIsNotNone(test_net)
        test_net = resnet.ResNet.resnet152(use_policy=True)
        self.assertIsNotNone(test_net)
    
    def test03_resnet34_noExceptionOnCreation(self):
        test_net = resnet.ResNet.resnet34()
        self.assertIsNotNone(test_net)
        test_net = resnet.ResNet.resnet34(use_policy=True)
        self.assertIsNotNone(test_net)
    
    def test04_setDropPolicy_setsCorrectPolicy(self):
        test_policy = resnet.ResNet.ResnetDropResidualPolicy()

        resnet.ResNet.setDropPolicy(test_policy)

        self.assertEqual(test_policy, resnet.ResNet.getDropPolicy())

    def test05_generateRandomNBoolArrayPermutations_returnsCorrectSizeAndAmount(self):
        max_size = 20
        max_true = 15
        
        test_array = resnet_utils.getRandomBoolListPermutation(max_size, max_true)

        self.assertEqual(len(test_array), max_size)
        self.assertEqual(sum(x == True for x in test_array), max_true)
        self.assertEqual(sum(x == False for x in test_array), max_size - max_true)
    
    def test06_resnet50_withDropRandNPolicy_noExceptionOnCreation(self):
        test_policy = resnet.DropPolicies.ResNetDropRandNPolicy(2)
        resnet.ResNet.setDropPolicy(test_policy)

        self.assertEqual(resnet.ResNet.getDropPolicy(), test_policy)

        test_net = resnet.ResNet.resnet50(use_policy=True)
        self.assertIsNotNone(test_net)
    
    def test07_resnet50_withDropRandNPolicy_noException_onForward(self):
        test_policy = resnet.DropPolicies.ResNetDropRandNPolicy(2)
        resnet.ResNet.setDropPolicy(test_policy)

        test_net = resnet.ResNet.resnet50(use_policy=True)
        test_net.forward(torch.rand(1, 3, 224, 224))