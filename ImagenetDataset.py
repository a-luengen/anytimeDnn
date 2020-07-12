
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from skimage import io
import numpy as np
import time
from PIL import Image


IMAGE_SIZE = (256, 256)
TRAIN_SPLIT_DEFAULT = 0.9

# source:
# https://gist.github.com/mf1024/a9199a325f9f7e98309b19eb820160a7
# Thanks mate!

class ImagenetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = TRAIN_SPLIT_DEFAULT, 
        random_seed = 42, target_transform = None, num_classes = None ):
        super(ImagenetDataset, self).__init__()

        self.data_path = data_path
        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes
        
        self.classes = []

        class_idx = 0

        for class_name in os.listdir(data_path):
            # skip accidental files in the data_path directory
            if not os.path.isdir(os.path.join(data_path, class_name)):
                continue
            
            self.classes.append(dict(class_idx = class_idx, class_name = class_name))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break
        
        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        # compute list of available images
        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name
                ))
        
        self.img_idxes = np.arange(0, len(self.image_list))

        # configure train or test images
        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]
        
    def __len__(self):
        return len(self.img_idxes)
    
    def __getitem__(self, index):
        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        # adapt grayscale image to match with 3-channel rgb
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        
        tr = transforms.ToTensor()
        img1 = tr(img)

        width, height = img.size
        if min(width, height) > IMAGE_SIZE[0] * 1.5:
            tr = transforms.Resize(int(IMAGE_SIZE[0] * 1.5))
            img = tr(img)
        
        width, height = img.size
        if min(width, height) < IMAGE_SIZE[0] * 1.5:
            tr = transforms.Resize(IMAGE_SIZE)
            img = tr(img)
        
        tr = transforms.RandomCrop(IMAGE_SIZE)
        img = tr(img)
        tr = transforms.ToTensor()
        img = tr(img)

        if img.shape[0] != 3:
            img = img[0:3]

        return dict(
            image = img, 
            cls = img_info['cls']['class_idx'], 
            class_name = img_info['cls']['class_name'])

def get_imagenet_datasets(data_path, random_seed = None, num_classes = None):
    
    if random_seed == None:
        random_seed = int(time.time())

    print(f"Random seed = {random_seed}")
    dataset_train = ImagenetDataset(data_path, is_train=True, random_seed=random_seed, num_classes = num_classes)
    dataset_test = ImagenetDataset(data_path, is_train=False, random_seed=random_seed, num_classes = num_classes)

    return dataset_train, dataset_test
