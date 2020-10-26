import unittest
import torch
from timeit import default_timer as timer

from .context import utils


class TestMsdNetInferencing(unittest.TestCase):


    def test000_forwardOnMsdNet4_NoExcpetion(self):

        model = utils.getModelWithOptimized('msdnet4')

        expected_outputs = 4
        output = model(torch.rand(1, 3, 224, 224))

        self.assertIsNotNone(output)
        self.assertEqual(expected_outputs, len(output))
    
    def test010_forwardOnMsdNet_NoException(self):

        model = utils.getModelWithOptimized('msdnet')

        expected_outputs = 5

        output = model(torch.rand(1, 3, 224, 224))

        self.assertIsNotNone(output)
        self.assertEqual(expected_outputs, len(output))
    
    def test020_forwardOnMSDNet4_ForAllMaxClassificationsPossible_ReturnsExactAmountOfOutputs(self):
        model = utils.getModelWithOptimized('msdnet4')
        max_classifications = [x for x in range(1, model.nBlocks + 1)]

        test_input = torch.rand(1, 3, 224, 224)

        for max_cls in max_classifications:
            model.setMaxClassifiers(max_cls)
            output = model(test_input)
            self.assertEqual(max_cls, len(output))
        
    def test030_forwardOnMsdNet_ForAllMaxClassificationsPossible_ReturnsExactAmountOfOutputs(self):
        model = utils.getModelWithOptimized('msdnet')

        max_classifications = [x for x in range(1, model.nBlocks + 1)]
        
        test_input = torch.rand(1, 3, 224, 224)

        for max_cls in max_classifications:
            model.setMaxClassifiers(max_cls)
            output = model(test_input)
            self.assertEqual(max_cls, len(output))
        
    def test040_forwardOnMsdNet_ShouldTakeLessTimeWithLessClassifications(self):

        model = utils.getModelWithOptimized('msdnet')

        max_classifications = [x for x in range(1, model.nBlocks + 1)]

        test_input = torch.rand(1, 3, 224, 224)

        time_results = []
        for max_cls in max_classifications:
            model.setMaxClassifiers(max_cls)
            start = timer()
            output = model(test_input)
            end = timer()
            time_results.append(end - start)
        for i in range(len(time_results) - 1):
            self.assertLessEqual(time_results[i], time_results[i + 1])