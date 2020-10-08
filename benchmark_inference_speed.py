import os
import shutil
import time
from timeit import default_timer as timer
import datetime
import sys
import logging
from tqdm import tqdm

from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
import densenet.densenet as dn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn import metrics


from utils import *
from data.ImagenetDataset import get_zipped_dataloaders
from data.utils import getLabelToClassMapping



AVAILABLE_STATE_DICTS = ['densenet121', 'densenet121-skip']# 'densenet169']
STATE_DICT_PATH = os.path.join(os.getcwd(), 'state')
BATCH_SIZE = 8

def executeSpeedBench():
    # run benchmark loop
    for i, arch in enumerate(AVAILABLE_STATE_DICTS):
        runSpeedBenchForModel(arch, warmup=True)
        runSpeedBenchForModel(arch, warmup=False)

def executeQualityEvaluation(only_best):
    total = len(AVAILABLE_STATE_DICTS)
    for i, arch in enumerate(AVAILABLE_STATE_DICTS):
        logging.info(f"Executing quality evaluation {i} of {total} for {arch}")
        runQualityBenchForModel(arch, only_best=only_best)

def runSpeedBenchForModel(arch: str, warmup=False):
    if not warmup:
        logging.info(f'###### Running Benchmark for {arch} network. ######')

    start_init = timer()
    model = getModelWithOptimized(arch)
    #logging.info(f'Time to init {arch} network: {timer() - start_init:.4f} seconds')

    model.eval()
    
    bench_input = torch.rand((1, 3, 224 ,224))
    start_inference = timer()
    output = model(bench_input)
    if not warmup:
        logging.info(f'Time for inferencing {arch} network: {timer() - start_inference:.6f} seconds')
        logging.info(f'####### Finished run #######')

def runQualityBenchForModel(arch: str, only_best=False)-> None:
    data_path = 'data/imagenet_full'
    #data_path = 'data/imagenet_red'

    model = getModelWithOptimized(arch)
    
    arch_states = [d for d in os.listdir(STATE_DICT_PATH) if arch.split("-")[0] in d]
    if only_best:
        arch_states = [d for d in arch_states if 'best' in d]

    model.eval()
    print(arch_states)

    _, _, val_loader =  get_zipped_dataloaders(data_path, BATCH_SIZE, use_valid=True)
    with torch.no_grad():
        for state in arch_states:
            model, _, epoch, prec = resumeFromPath(os.path.join(STATE_DICT_PATH, state), model)
            logging.info(f"Resuming {state} from epoch {epoch} with best precision {prec}...")
            classes = getLabelToClassMapping(os.path.join(os.getcwd(), data_path))

            grndT, pred = evaluateModel(model, val_loader, classes)
            logging.info(metrics.classification_report(grndT, pred, digits=3))
            generateAndStoreClassificationReportCSV(grndT, pred, f'{arch}_report.csv')

def evaluateModel(model, loader, classes): 
    pred, grndT = [], []
    with tqdm(total=len(loader), desc="Progress", ncols=100) as pbar:
        for (images, labels) in iter(loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pred = pred + [classes[predicted[k]] for k in range(min(BATCH_SIZE, labels.shape[0]))]
            grndT = grndT + [classes[labels[j]] for j in range(min(BATCH_SIZE, labels.shape[0]))]
            pbar.update(1)
    return grndT, pred

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    

    executeSpeedBench()
    #executeQualityEvaluation(True)