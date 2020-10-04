import unittest
import os
import shutil
import random
import torchvision.models as models

from .context import data_utils as du
#from data_utils import *

class TestUtilFunctions(unittest.TestCase):

    image_test_path = "data/test-images"
    
    

    image_base_path = os.path.join(os.getcwd(), image_test_path)

    test_new_dataset_path = os.path.join(image_base_path, 'test-set')
    image_src_train_path = os.path.join(image_base_path, 'test-train')
    image_src_val_path = os.path.join(image_base_path, 'test-val')
    image_tar_train_path = os.path.join(image_base_path, "train")
    image_tar_val_path = os.path.join(image_base_path, "val")

    def setUp(self):
        path = self.image_tar_train_path
        if not os.path.exists(path):
            shutil.copytree(self.image_src_train_path, path)
        
        path = self.image_tar_val_path
        if not os.path.exists(path):
            shutil.copytree(self.image_src_val_path, path)

    def tearDown(self):
        if os.path.exists(self.image_tar_train_path):
            shutil.rmtree(self.image_tar_train_path)
        if os.path.exists(self.image_tar_val_path):
            shutil.rmtree(self.image_tar_val_path)    
        if os.path.exists(self.test_new_dataset_path):
            shutil.rmtree(self.test_new_dataset_path)    

    def test00_foldersWithTestContentExist(self):
        self.assertTrue(os.path.exists(self.image_base_path))
        self.assertTrue(os.path.exists(self.image_src_train_path))
        self.assertTrue(os.path.exists(self.image_src_val_path))

    def test01_transformAllImages_noException(self):
        du.transformAllImages(self.image_tar_val_path, 'val')
        du.transformAllImages(self.image_tar_train_path, 'train')
    
    def test02_processImagesByRatio_noException(self):

        def isJPG(string:str)->bool:
            return '.jpg' in string.casefold() or '.png' in string.casefold()

        img_count_src = len(list(filter(isJPG, os.listdir(self.image_src_train_path))))

        du.processImagesByRatio(1.0, self.image_src_train_path, self.image_tar_train_path, 'train')
        
        img_count_tar = len(list(filter(isJPG, os.listdir(self.image_tar_train_path))))
        
        self.assertEqual(img_count_src, img_count_tar)

    def test03_generateNewImageDataset_noException(self):
        du.generateNewImageDataset(self.image_base_path, self.test_new_dataset_path, 'val', ratio=1.0)
    
    def test04_assertRatioCalculationIsCorrect(self):
        
        src = len(os.listdir(self.image_tar_train_path))

        ratio = 1.0

        tar = int(src * ratio)

        self.assertEqual(src, tar)
