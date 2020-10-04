import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dareblopy as db
from IPython.display import Image, display

import os
import shutil
import time

from skimage import io
import numpy as np
import time
import zipfile
from PIL import Image

def processImagesByRatio(ratio: int, src_path : str, tar_path : str, set_type : str):
    src_path = os.path.join(src_path, set_type)

    if not os.path.exists(tar_path):
        os.mkdir(tar_path)

    if not os.path.isdir(tar_path):
        os.mkdir(tar_path)
    tar_path = os.path.join(tar_path, set_type)

    if not os.path.isdir(src_path):
        print(f"No source path found for {src_path}")
        return

    if not os.path.isdir(tar_path):
        print(f"creating target path at {tar_path}")
        os.mkdir(tar_path)

    val_cl_src_paths = []

    def isFolder(fn:str)->bool:
        return not '.jpg' in fn

    classes = list(filter(isFolder, os.listdir(src_path)))
    for val_class in classes:
        val_cl_src_paths.append(os.path.join(src_path, val_class))
    
    print(f"val classes: {len(val_cl_src_paths)}")
    
    # index-class mapping file
    filename = "index-" + set_type + ".txt"
    with open(os.path.join(tar_path, filename), 'w') as fileHandle:
        # copy all images from set
        index = 0
        img_name = ""
        for class_name, src_path in zip(classes, val_cl_src_paths):
            class_img_paths = os.listdir(src_path)
            total_amount = len(class_img_paths)
            cp_amount = int(ratio * total_amount)

            print(f"Copy {cp_amount}/{total_amount} to {tar_path}")

            for i in range(0, cp_amount):
                if not os.path.isdir(tar_path):
                    os.mkdir(tar_path)

                file_type = class_img_paths[i].split(".")[1]
                # remove any wired URL encoded parts from the filetype
                if len("jpg") < len(file_type): 
                    file_type = file_type[:3]
                img_name = f"{index}.{file_type}"

                cp_from = os.path.join(src_path, class_img_paths[i])
                cp_to = os.path.join(tar_path, img_name)
                fileHandle.write(class_name + "\n")
                print(f"Copy: \n --{cp_from} \n" 
                        + f" ->{cp_to}")

                shutil.copyfile(cp_from, cp_to)
                
                index = index + 1
        return fileHandle.name
    return None

def getClassToIndexMapping(path: str):
    print(path)
    if not os.path.isfile(path):
        raise Exception(f"No Index file found at location: {path}")
    mapping = []
    file = open(path, 'r')
    for line in file:
        mapping.append(line.replace("\n", ""))
    file.close()
    return mapping

def getLabelToClassMapping(data_path: str):
    index_path = os.path.join(data_path, 'index-val.txt')

    if not os.path.isfile(index_path):
        raise Exception(f"No index file found in {data_path}")

    mapping = getClassToIndexMapping(index_path)
    label_to_class = list(set(mapping))
    label_to_class.sort()
    return label_to_class

def transformTrainImage(img: Image) -> Image:
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return trans(img)

def transformValImage(img: Image) -> Image:
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return trans(img)

def transformAllImages(path: str, set_type: str) -> None:

    if path is None or not os.path.isdir(path):
        raise Exception(f"No path available for source: {path}")
    
    if set_type == "val":
        transform = transformValImage

    elif set_type == "train":
        transform = transformTrainImage
    else:
        return

    def isImg(fn:str)->bool:
        return '.jpg' in fn.casefold()

    img_name_list = list(filter(isImg, os.listdir(path)))

    for img_name in img_name_list:
        img_path = os.path.join(path, img_name)
        with Image.open(img_path) as img:
            if img.mode == 'L':
                img = transforms.Grayscale(num_output_channels=3)(img)
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            try:
                img = transform(img)
            except Exception as e:
                raise Exception(f"{e}\n{img_name}")
            
            img = transforms.ToPILImage()(img)
            img.save(img_path)

def generateDatasetZipArchive(base_path: str, files_path: str, prefix: str, set_type: str) -> None:    
    filenames = os.listdir(files_path)
    zipPath = os.path.join(base_path, prefix + set_type + '.zip')
    if os.path.isfile(zipPath):
        os.remove(zipPath)
    
    with zipfile.ZipFile(zipPath, mode='w') as zipArch:
        for filename in filenames:
            zipArch.write(os.path.join(files_path, filename), arcname=filename)

def generateNewImageDataset(from_base: str, to_base: str, set_type: str, ratio=1/8, force_process=True) -> None:
    
    if not (set_type == 'val' or set_type == 'train'):
        raise Exception(f'{set_type} is not supported')
    
    target_path = os.path.join(to_base, set_type)
    # clean up to_path
    if force_process and os.path.isdir(target_path):
        print(f"Removing existing directory: {target_path}")
        shutil.rmtree(target_path)
        os.remove(os.path.join(to_base, f"index-{set_type}.txt"))
    
    if force_process:
        processImagesByRatio(ratio, from_base, to_base, set_type)
    transformAllImages(os.path.join(to_base, set_type), set_type)
    generateDatasetZipArchive(to_base, os.path.join(to_base, set_type), 'index-', set_type)

def benchmarkDataloader(loader: DataLoader, it_limit=None) -> (float, float, float):
    batch_size = loader.batch_size
    avg = 0.0
    maxTime = 0.0
    minTime = 100000.0
    start = time.time()
    for i, (img, label) in enumerate(loader):
        stop = time.time() - start
        if (stop > maxTime): maxTime = stop
        elif (stop < minTime): minTime = stop
        if it_limit is not None and i == it_limit - 1:
            print(i, flush=True)
            break
        avg += stop
        start = time.time()
    
    avg = (avg / (min(len(loader), float(it_limit))))
    print(f"Avg batch load time {avg * 1000:8.2f} ms - Max: {maxTime * 1000:8.2f}ms - Min: {minTime * 1000:8.2f}ms")
    avg /= float(batch_size)
    maxTime /= float(batch_size)
    minTime /= float(batch_size)
    print(f"Avg image load time {avg * 1000:8.2f} ms - Max: {maxTime * 1000:8.2f}ms - Min: {minTime * 1000:8.2f}ms")
    return (avg, maxTime, minTime)