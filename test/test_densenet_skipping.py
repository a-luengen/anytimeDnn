import unittest
import torch
import torchvision
from .context import resnet
from .context import resnet_utils
from .context import densenet
from .context import utils
from .context import data_loader

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

    def test020_densenet121_withSkipping_sameParameterAmount(self):
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
    
    def test040_DNDropRandNPolicy_initWithoutException(self):
        test_block_config = (1, 1, 1)

        test_policy = densenet.DropPolicies.DenseNetDropPolicy(test_block_config)
        self.assertIsNotNone(test_policy)
    
    def test045_splitListIn2DList(self):
        list_base = (3, 4, 5)
        list_a = resnet_utils.getRandomBoolListPermutation(sum(list_base), 5)

        list_res = []

        split_list = []
        prev_val = 0    
        for x in list_base:
            split_list.append((prev_val, prev_val + x))
            prev_val = prev_val + x
        
        list_res = [list_a[i:j] for i, j in split_list]

        for l, r in zip(list_res, list_base):
            self.assertEqual(len(l), r)

    def test050_DNDropRandNPolicy_getLayerConfiguration_ReturnsCorrectAmountOfElementsPerLayer(self):
        test_block_config = (3, 4, 5)
        test_n = 4
        test_policy = densenet.DropPolicies.DNDropRandNPolicy(test_block_config, test_n)

        for i, count in enumerate(test_block_config):
            layer_config = test_policy.getDropLayerConfiguration(i)
            self.assertEqual(len(layer_config), count)
    
    def test060_DNDropRandNPolicy_ConfigurationContainsExactNTrueValues(self):
        test_block_config = (4, 3, 2)
        test_n = 5
        test_policy = densenet.DropPolicies.DNDropRandNPolicy(test_block_config, test_n)

        l_of_l = [test_policy.getDropLayerConfiguration(i) for i in range(len(test_block_config))]

        # flatten list
        flat_list = [item for l in l_of_l for item in l]

        self.assertEqual(len(flat_list), sum(test_block_config))
        self.assertEqual(sum(flat_list), test_n)

    def test065_GetSkipPolicy_ReturnsPreviouslySetPolicy(self):
        test_pol = densenet.DropPolicies.DenseNetDropPolicy(None)
        densenet.DropPolicies.setSkipPolicy(test_pol)
        test_res = densenet.DropPolicies.getSkipPolicy()
        self.assertEqual(test_pol, test_res)

    def test070_DenseNetWithoutBlockConfig_SameResultAsPlainResnet(self):
        test_policy = densenet.DropPolicies.DNDropRandNPolicy((6, 12, 24, 16), 0)
        densenet.DropPolicies.setSkipPolicy(test_policy)

        skip_net = densenet.torchDensenet.densenet121(num_classes=40, use_skipping=True)
        plain_net = utils.getModel('densenet121')
        plain_net.load_state_dict(skip_net.state_dict())

        with torch.no_grad():
            skip_net.eval()
            plain_net.eval()

            _, loader, _ = data_loader.get_zipped_dataloaders(data_loader.REDUCED_SET_PATH, 1)

            (img, target) = next(iter(loader))

            res_skip = skip_net(img)
            res_plain = plain_net(img)

            self.assertTrue(torch.equal(res_skip, res_plain))