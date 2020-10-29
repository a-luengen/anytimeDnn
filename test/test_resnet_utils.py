import unittest

from .context import resnet_utils

class TestResnetUtils(unittest.TestCase):


    def test000_getRandomBoolPermutation_shouldReturnCorrectSize(self):

        test_size = 20
        test_max_n = 4

        random_perm = resnet_utils.getRandomBoolListPermutation(test_size, test_max_n)

        self.assertEqual(test_size, len(random_perm))
    
    def test010_getRandomBoolPermutation_hasExactlyNTrueValues(self):

        test_size = 30
        test_max_n = 13

        random_perm = resnet_utils.getRandomBoolListPermutation(test_size, test_max_n)

        self.assertEqual(test_max_n, sum(random_perm))
    
    def test020_getGaussDistributedBoolList_HasExactlyNTrueValues(self):

        test_size = 30
        test_max_n = 12

        res_bool_list = resnet_utils.getGaussDistributedBoolList(test_size, test_max_n)

        self.assertEqual(test_max_n, sum(res_bool_list))