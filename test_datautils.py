import unittest
import os
import shutil
import random
import torchvision.models as models
from data.utils import *

class TestUtilFunctions(unittest.TestCase):

    image_test_path = "data/test-images"

    image_base_path = os.path.join(os.getcwd(), image_test_path)
    image_src_train_path = os.path.join(image_base_path, "train")
    image_src_val_path = os.path.join(image_base_path, "val")
    image_tar_train_path = os.path.join(image_base_path, 'test-train')
    image_tar_val_path = os.path.join(image_base_path, 'test-val')

    def setupClass(self):
        if not os.path.exists(self.image_tar_train_path):
            os.mkdir(self.image_tar_train_path)
        if not os.path.exists(self.image_tar_val_path):
            os.mkdir(self.image_tar_val_path)

    def test00_foldersWithTestContentExist(self):
        self.assertTrue(os.path.exists(self.image_base_path))
        self.assertTrue(os.path.exists(self.image_src_train_path))
        self.assertTrue(os.path.exists(self.image_src_val_path))

    def test01_transformAllImages_noException(self):
        
        transformAllImages(self.image_test_path, 'val')