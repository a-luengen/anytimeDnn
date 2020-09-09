import unittest
import os
import shutil
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
    
    def test01_testCheckpointFunction_withIsBestTrue_noException(self):

        state_dict = getStateDict(self.test_net, 0, self.arch_name, 0.0, self.test_optim)

        save_checkpoint(state_dict, True, self.arch_name, self.test_dir)