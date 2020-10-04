import unittest

from .context import resnet

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

