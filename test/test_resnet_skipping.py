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

        resnet.DropPolicies.setDropPolicy(test_policy)

        test_net = resnet.ResNet.resnet18(use_policy=True)

        test_net.forward(torch.rand(1, 3, 224, 224))

        self.assertEqual( sum(test_policy.getDropConfig()) , test_n )
    
    def test09_getModelWithOptimized_returnsResNet18_policyIsSet(self):
        test_n = 3
        model = utils.getModelWithOptimized('resnet18-drop-rand-n', n=test_n, batch_size=10)
        res_policy = resnet.DropPolicies.getDropPolicy()
        self.assertIsNotNone(model)
        self.assertIsNotNone(res_policy)
        self.assertTrue(isinstance(res_policy, resnet.DropPolicies.ResNetDropRandNPolicy))
        self.assertEqual(sum(res_policy.getDropConfig()), test_n)
    
    def test095_getModelWithOptimized_returnsResNet34_AndPolicyIsSet(self):
        test_n = 5
        model = utils.getModelWithOptimized('resnet34-drop-rand-n', n=test_n, batch_size=10)

        res_policy = resnet.DropPolicies.getDropPolicy()

        self.assertIsNotNone(model)
        self.assertIsNotNone(res_policy)
        self.assertIsInstance(res_policy, resnet.DropPolicies.ResNetDropRandNPolicy)
        self.assertEqual(sum(res_policy.getDropConfig()), test_n)
    
    def test097_getModelWithOptimized_returnsResNet18_AndDropLastRandNPolicyIsSet(self):
        test_n = 2
        model = utils.getModelWithOptimized('resnet18-drop-last-rand-n', n=test_n, batch_size=10)

        res_policy = resnet.DropPolicies.getDropPolicy()

        self.assertIsNotNone(model)
        self.assertIsNotNone(res_policy)
        self.assertIsInstance(res_policy, resnet.DropPolicies.ResNetDropRandLastNPolicy)
        self.assertEqual(sum(res_policy.getDropConfig()), test_n)
    

    def test100_resnet18_withDropRandNPolicy_Forward2Times_noException(self):
        test_n = 3

        model = utils.getModelWithOptimized('resnet18-drop-rand-n', n=test_n)

        test_input = torch.rand(1, 3, 224, 224)
        model(test_input)
        model(test_input)
        pass

    def test110_resnetDropLastRandNLayers_hasCorrectAmountOfTrueValuesInConfig(self):

        test_n = 4
        test_max_layers = 10

        policy = resnet.DropPolicies.ResNetDropRandLastNPolicy(test_n)
        policy.setMaxSkipableLayers(test_max_layers)

        config_list = policy.getDropConfig()

        self.assertEqual(test_n, sum(config_list))
    
    def test120_resnetDropLastRandNLayers_hasOnlyLastLayersSetToTrue(self):

        test_n = 4
        test_max_layers = 10

        policy = resnet.DropPolicies.ResNetDropRandLastNPolicy(test_n)

        policy.setMaxSkipableLayers(test_max_layers)

        config_list = policy.getDropConfig()

        false_list = config_list[0:test_max_layers - 4]
        true_list = config_list[test_max_layers - 4:test_max_layers]

        self.assertEqual(sum(false_list), 0)
        self.assertEqual(len(false_list), test_max_layers - 4)
        self.assertEqual(sum(true_list), test_n)
        self.assertEqual(len(true_list), test_n)
    
    def test130_resnetDropNRandNormalDistributionPolicy_IsSet_OnWithOptimized(self):

        test_arch = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

        test_policy = 'drop-norm-n'

        test_n = 5
        test_batch = 20

        for arch in test_arch:
            model = utils.getModelWithOptimized(arch + '-' + test_policy, test_n, test_batch)
            res_policy = resnet.DropPolicies.getDropPolicy()

            self.assertIsNotNone(res_policy)
            self.assertIsInstance(res_policy, resnet.DropPolicies.ResNetDropNRandNormalDistributionPolicy)