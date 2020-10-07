import unittest
import torch
import torchvision
from .context import resnet
from .context import resnet_utils
from .context import densenet
from .context import utils

class TestDenseNetSkippingPolicies(unittest.TestCase):

    def getParameterCountList(self, model):
        return [p.data.nelement() for p in model.parameters()]

    def test000_densnet121_withoutSkipping_noException_onForward(self):
        model = utils.getModel('densenet121')
        self.assertIsNotNone(model)
        test_result = model(torch.rand(1, 3, 224, 224))
        self.assertIsNotNone(test_result)
    
    def test010_densenet121_withoutSkipping_sameParameters_asOriginalTorchImpl(self):
        densenet121 = utils.getModel('densenet121')
        num_param_1 = self.getParameterCountList(densenet121)

        torch_model = torchvision.models.densenet121(num_classes=40)
        num_param_2 = self.getParameterCountList(torch_model)

        for i, (res_count, torch_count) in enumerate(zip(num_param_1, num_param_2)):
            self.assertEqual(res_count, torch_count, f"Found in index {i}")

        self.assertEqual(sum(num_param_1), sum(num_param_2))

    def test020_densenet121_withSkipping_sameParameters(self):
        skipnet = utils.getModelWithOptimized('densenet121-skip')
        num_param = self.getParameterCountList(skipnet)

        torch_model = torchvision.models.densenet121(num_classes=40)
        num_param2 = self.getParameterCountList(torch_model)

        self.assertEqual(sum(num_param), sum(num_param2))
    
    def test030_densenet121_withSkipping_noExceptionOnForward(self):
        skipnet = utils.getModelWithOptimized('densenet121-skip')
        test_t = torch.rand(1, 3, 224, 224)

        skipnet(test_t)
        pass
    
