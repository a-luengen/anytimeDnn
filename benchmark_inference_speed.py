import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import shutil
import time
import datetime
import sys
import logging
import cProfile
from tqdm import tqdm
from timeit import default_timer as timer
from sklearn import metrics

import densenet.densenet as dn
from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
from utils import *
from data.ImagenetDataset import get_zipped_dataloaders
from data.utils import getLabelToClassMapping

#AVAILABLE_STATE_DICTS = ['densenet121', 'densenet121-skip']# 'densenet169']
#AVAILABLE_STATE_DICTS = ['densenet121-skip']# 'densenet169']
AVAILABLE_STATE_DICTS = ['resnet18-drop-rand-n', 'resnet34-drop-rand-n', 'resnet50-drop-rand-n']
AVAILABLE_STATE_DICTS = ['resnet50-drop-rand-n']
AVAILABLE_STATE_DICTS = ['resnet34-drop-rand-n']
AVAILABLE_STATE_DICTS = ['densenet121']
#AVAILABLE_STATE_DICTS = ['resnet18', 'resnet34', 'resnet50',]
#AVAILABLE_STATE_DICTS = ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet169']
STATE_DICT_PATH = os.path.join(os.getcwd(), 'state')
BATCH_SIZE = 1
SPEED_RUNS = 30
QUALITY_RUNS = 10
LAYERS_TO_SKIP = 0
DATA_PATH = 'data/imagenet_red'

def executeSpeedBench():
    # run benchmark loop
    with tqdm(total=(len(AVAILABLE_STATE_DICTS) * SPEED_RUNS), ncols=100, desc="Speed Progress") as pbar:

        for _, arch in enumerate(AVAILABLE_STATE_DICTS):
            logging.info(f'###### Running Benchmark for {arch} network. ######')
            measurements = []
            for _ in range(0, SPEED_RUNS):
                measurements.append(
                    runSpeedBenchForModel(arch, warmup=True)
                )
                pbar.update(1)
            print()
            print(measurements)
            print(f'Avg.: {sum(measurements) / len(measurements):.4f}')
            print()


def runSpeedBenchForModel(arch: str, warmup=False) -> float:   

    #start_init = timer()
    model = getModelWithOptimized(arch, n=LAYERS_TO_SKIP)
    #logging.info(f'Time to init {arch} network: {timer() - start_init:.4f} seconds')

    model.eval()
    
    bench_input = torch.rand((1, 3, 224 ,224))

    if warmup:
        #logging.info(f'##### Running warmup first #####')
        output = model(bench_input)

    start_inference = timer()

    output = model(bench_input)
    end_inference = timer()
    #logging.info(f'Time for inferencing {arch} network: {end_inference - start_inference:.6f} seconds')
    #logging.info(f'####### Finished run #######')

    return end_inference - start_inference

def executeQualityEvaluation(only_best):
    total = len(AVAILABLE_STATE_DICTS)
    for i, arch in enumerate(AVAILABLE_STATE_DICTS):
        logging.info(f"Executing quality evaluation {i + 1} of {total} for {arch}")
        runQualityBenchForModel(arch, only_best=only_best)

def runQualityBenchForModel(arch: str, only_best=False, persist_results=False)-> None:
    #data_path = 'data/imagenet_full'
    data_path = DATA_PATH

    arch_states = [d for d in os.listdir(STATE_DICT_PATH) if arch.split("-")[0] in d]
    if only_best:
        arch_states = [d for d in arch_states if 'best' in d]
    
    print(arch_states)

    _, _, val_loader =  get_zipped_dataloaders(data_path, BATCH_SIZE, use_valid=True)
    with torch.no_grad():
        for state in arch_states:
            measurements = []
            for _ in range(0, QUALITY_RUNS):
                model = getModelWithOptimized(arch, n=LAYERS_TO_SKIP, batch_size=BATCH_SIZE)
                model.eval()
                model, _, _, _ = resumeFromPath(os.path.join(STATE_DICT_PATH, state), model)
                #logging.info(f"Resuming {state} from epoch {epoch} with best precision {prec}...")
                classes = getLabelToClassMapping(os.path.join(os.getcwd(), data_path))

                grndT, pred = evaluateModel(model, val_loader, classes, BATCH_SIZE)
                measurements.append(metrics.accuracy_score(grndT, pred))
                if persist_results:
                    logging.info(metrics.classification_report(grndT, pred, digits=3))
                    generateAndStoreClassificationReportCSV(grndT, pred, f'{arch}_report.csv')
                logging.debug(measurements)
            logging.info('')
            logging.info(measurements)
            logging.info(f'Precission: {sum(measurements) / len(measurements):.4f}')    

def evaluateModel(model, loader, classes, batch_size): 
    pred, grndT = [], []
    with tqdm(total=len(loader), desc="Progress", ncols=100) as pbar:
        for (images, labels) in iter(loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pred = pred + [classes[predicted[k]] for k in range(min(batch_size, labels.shape[0]))]
            grndT = grndT + [classes[labels[j]] for j in range(min(batch_size, labels.shape[0]))]
            pbar.update(1)
    return grndT, pred

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    

    #executeSpeedBench()
    executeQualityEvaluation(True)
    #cProfile.run('executeSpeedBench()')
    
