import unittest
import torch
import torchvision
from .context import resnet
from .context import resnet_utils
from .context import utils

class TestResnetSkippingPolicies(unittest.TestCase):

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
    
    def test08_resnet18_withDropRandNPolicy_noException_onForward(self):
        test_n = 1
        test_policy = resnet.DropPolicies.ResNetDropRandNPolicy(test_n)

        resnet.ResNet.setDropPolicy(test_policy)

        test_net = resnet.ResNet.resnet18(use_policy=True)

        test_net.forward(torch.rand(1, 3, 224, 224))

        self.assertEqual( sum(test_policy.getDropConfig()) , test_n )
    
    def test09_getModelWithOptimized_returnsResNet18_policyIsSet(self):
        test_n = 3
        model = utils.getModelWithOptimized('resnet18-drop-rand-n', n=test_n)

        self.assertIsNotNone(model)
        self.assertIsNotNone(resnet.ResNet.getDropPolicy())
        self.assertTrue(isinstance(resnet.ResNet.getDropPolicy(), resnet.DropPolicies.ResNetDropRandNPolicy))
        self.assertEqual(sum(resnet.ResNet.getDropPolicy().getDropConfig()), test_n)

    def test10_resnet18_withDropRandNPolicy_Forward2Times_noException(self):
        test_n = 3

        model = utils.getModelWithOptimized('resnet18-drop-rand-n', n=test_n)

        test_input = torch.rand(1, 3, 224, 224)
        model(test_input)
        model(test_input)
        pass