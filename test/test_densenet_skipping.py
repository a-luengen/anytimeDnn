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
        skipnet = utils.getModelWithOptimized('densenet121-skip', 1, 1)
        num_param = self.getParameterCountList(skipnet)

        torch_model = torchvision.models.densenet121(num_classes=40)
        num_param2 = self.getParameterCountList(torch_model)

        self.assertEqual(sum(num_param), sum(num_param2))
    
    def test030_densenet121_withSkipping_noExceptionOnForward(self):
        skipnet = utils.getModelWithOptimized('densenet121-skip', 1, 1)
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

    def test067_GetSkipPolicy_ReturnsPreviouslySetPolicy_WithExactConfiguration(self):
        test_block_config = (6, 12, 24, 16)
        test_n = 10
        test_pol = densenet.DropPolicies.DNDropRandNPolicy(test_block_config, test_n)
        densenet.DropPolicies.setSkipPolicy(test_pol)

        ret_pol = densenet.DropPolicies.getSkipPolicy()

        self.assertEqual(ret_pol.getFullConfig, test_pol.getFullConfig)

    def test068_DNDropRandNPolicy_WihtMoreNThanPossible_ShouldThrowException(self):
        test_block_config = (2, 3, 4)
        test_n = 10

        self.assertRaises(
            ValueError, 
            densenet.DropPolicies.DNDropRandNPolicy,
            test_block_config, 
            test_n
            )

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
        
    def test080_DNDropLastNPolicy_hasOnlyLastBlockConfigEntries_setToTrue(self):
        test_block_config = (5, 3, 2)
        test_n = 5

        policy_under_test = densenet.DropPolicies.DNDropLastNPolicy(test_block_config, test_n)

        res_full_config = policy_under_test.getFullConfig()

        exp_full_config = [sorted(x) for x in res_full_config]
        temp = []
        for ls in res_full_config:
            temp += ls

        self.assertEqual(sum(temp), test_n)
        self.assertEqual(exp_full_config, res_full_config)
    
    def test090_DenseNet169_WithDNDropLastNPolicy_NoExceptionOnForward(self):
        test_n = 3
        test_batch = 8

        test_net = utils.getModelWithOptimized('densenet169-skip-last', test_n, test_batch)

        test_tensor = torch.rand(test_batch, 3, 224, 224)

        output = test_net(test_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], test_batch)
    
    def test100_DenseNet169_WithDNDropLastNPolicy_MaximumN_NoExceptionOnForward(self):
        max_n = 72
        test_batch = 8
        
        test_net = utils.getModelWithOptimized('densenet169-skip-last', max_n, test_batch)

        test_tensor = torch.rand(test_batch, 3, 224, 224)

        output = test_net(test_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], test_batch)
    
    def test110_DenseNet121_WithDNDropLastNPolicy_MaximumN_NoExceptionOnForward(self):
        max_n = 58
        test_batch = 8

        test_net = utils.getModelWithOptimized('densenet121-skip-last', max_n, test_batch)

        test_tensor = torch.rand(test_batch, 3, 224, 224)
        
        output = test_net(test_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], test_batch)
    
    def test120_DenseNet121_WithDropRandNPolicy_MaximumN_NoExceptionOnForward(self):
        max_n = 58
        test_batch = 8

        test_net = utils.getModelWithOptimized('densenet121-skip', max_n, test_batch)

        output = test_net(torch.rand(test_batch, 3, 224, 224))

        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], test_batch)
    
    def test130_ForAllDenseNetArchs_SkipPolicyIsSet_OnGetModelWithOptimized(self):
        archs = ['densenet121', 'densenet169']

        test_n = 9
        test_batch = 9

        for arch in archs:
            utils.getModelWithOptimized(arch + '-skip', n=test_n, batch_size=test_batch)
            policy = densenet.DropPolicies.getSkipPolicy()

            self.assertIsNotNone(policy)
            self.assertIsInstance(policy, densenet.DropPolicies.DNDropRandNPolicy)

    def test140_ForAllDenseNetArchs_SkipLastNPolicyIsSet_OnGetModelWithOptimized(self):

        archs = ['densenet121', 'densenet169']
        test_n = 9
        test_batch = 10

        for arch in archs:
            utils.getModelWithOptimized(arch + '-skip-last', n=test_n, batch_size=test_batch)
            policy = densenet.DropPolicies.getSkipPolicy()

            self.assertIsNotNone(policy)
            self.assertIsInstance(policy, densenet.DropPolicies.DNDropLastNPolicy)
    
    def test150_DNDropLastNOfEachBlockPolicy_HasCorrectAmountOfSkips(self):

        test_block_config = (3, 3, 3, 3)
        test_n = 4

        test_policy = densenet.DropPolicies.DNDropLastNOfEachBlockPolicy(test_block_config, test_n)

        resulting_skips = 0
        for i in range(len(test_block_config)):
            resulting_skips += sum(test_policy.getDropLayerConfiguration(i))
        
        self.assertIsNotNone(test_policy)
        self.assertNotEqual(0, len(test_policy.getFullConfig()))
        self.assertEqual(resulting_skips, test_n)
        
    def test160_DNDropLastNOfEachBlockPolicy_HasCorrectAmountOfSkipsPerLayer(self):
        test_block_config = (3, 3, 3, 3, 3)
        test_n = 10

        expected_skips_per_layer = test_n // len(test_block_config)

        test_policy = densenet.DropPolicies.DNDropLastNOfEachBlockPolicy(test_block_config, test_n)

        for i in range(len(test_block_config)):
            self.assertEqual(sum(test_policy.getDropLayerConfiguration(i)), expected_skips_per_layer)
    
    def test170_DNDropLastNOfEachBlockPolicy_HasCorrectAmountOfSkipsPerLayer(self):
        test_block_config = (3, 1, 3, 1)
        test_n = 6
        expected_skips_per_layer = [2, 1, 2, 1]

        test_policy = densenet.DropPolicies.DNDropLastNOfEachBlockPolicy(test_block_config, test_n)

        for i, amount in enumerate(expected_skips_per_layer):
            self.assertEqual(sum(test_policy.getDropLayerConfiguration(i)), amount)
        
    def test180_DenseNet121_WithDNDropLastNOfEachBlockPolicy_NoExceptionOnForward_WithMaxBlocksToDrop(self):
        
        block_config = (6, 12, 24, 16)

        test_n, test_batch = sum(block_config), 1

        test_tensor = torch.rand(test_batch, 3, 224, 224)

        test_net = utils.getModelWithOptimized('densenet121-skip-last-n-block', test_n, test_batch)

        test_net(test_tensor)
    
    def test190_DenseNet169_WithDNDropLastNOfEachBlockPolicy_NoExceptionOnForward_WithMaxBlocksToDrop(self):
        block_config = (6, 12, 32, 32)

        test_n, test_batch = sum(block_config), 3

        test_tensor = torch.rand(test_batch, 3, 224, 224)

        test_net = utils.getModelWithOptimized('densenet169-skip-last-n-block', test_n, test_batch)

        test_net(test_tensor)

        result_skip_sum = 0
        for i in range(len(block_config)):
            result_skip_sum += sum(densenet.DropPolicies.getSkipPolicy().getDropLayerConfiguration(i))

        self.assertEqual(result_skip_sum, test_n)
    
    def test200_DNDropNormalDistributedN_ExactlyNLayersToDrop(self):

        test_block_config = (4, 10, 2, 3)
        test_n = 10

        test_policy = densenet.DropPolicies.DNDropNormalDistributedN(test_block_config, test_n)

        res_drop = 0
        for i in range(len(test_block_config)):
            layer_conf = test_policy.getDropLayerConfiguration(i)
            self.assertEqual(len(layer_conf), test_block_config[i])
            res_drop += sum(layer_conf)
        
        self.assertEqual(test_n, res_drop)