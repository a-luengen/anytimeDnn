
import os
import random
import logging
import torch
import argparse
import pandas as pd
import traceback
from tqdm import tqdm
from utils import getModelWithOptimized, resumeFromPath
from timeit import default_timer as timer
from data.utils import getLabelToClassMapping
from data.ImagenetDataset import get_zipped_dataloaders, REDUCED_SET_PATH
from sklearn import metrics
from collections import OrderedDict

STATE_DICT_PATH = os.path.join(os.getcwd(), 'state')

parser = argparse.ArgumentParser(description='Train several image classification network architectures.')
parser.add_argument('--batch_size', metavar='N', type=int, default=argparse.SUPPRESS, help='Batchsize for training or validation run.')
parser.add_argument('--runs', metavar='N', type=int, default=30, help='Number of runs to collect data for each item to benchmark.')
parser.add_argument('--only_arch', type=str, default=None, choices=['resnet', 'densenet'], help='Only benchmark architectures as specified by given string.')
parser.add_argument('--bench_type', type=str, default=None, choices=['quality', 'speed'], help='Execute only the specfied benchmark type.')

def evaluateModel(model, loader, classes, batch_size): 
    pred, grndT = [], []
    for (images, labels) in iter(loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        pred = pred + [classes[predicted[k]] for k in range(min(batch_size, labels.shape[0]))]
        grndT = grndT + [classes[labels[j]] for j in range(min(batch_size, labels.shape[0]))]
    return grndT, pred

def executeQualityBench(arch_name: str, loader, skip_n: int, classes, batch_size: int):
    
    model = getModelWithOptimized(arch_name, n=skip_n, batch_size=batch_size)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    arch = arch_name.split('-')[0]
    model, _, _, _ = resumeFromPath(os.path.join(STATE_DICT_PATH, f'{arch}_model_best.pth.tar'), model)

    grndT, pred = evaluateModel(model, loader, classes, batch_size)

    logging.debug(metrics.classification_report(grndT, pred, digits=3))
    acc = metrics.accuracy_score(grndT, pred)
    prec = metrics.precision_score(grndT, pred, average='macro')
    rec = metrics.recall_score(grndT, pred, average='macro')
    f1 = metrics.f1_score(grndT, pred, average='macro')
    return acc, prec, rec, f1

def executeSpeedBench(arch_name:str, skip_n:int):
    model = getModelWithOptimized(arch_name, n=skip_n, batch_size=1)
    model.eval()

    bench_input = torch.rand((1, 3, 224, 224))
    output = model(bench_input)

    start = timer()
    output = model(bench_input)
    stop = timer()

    return stop - start

def storeReportToCSV(reports_path:str, filename:str, data):
    df = pd.DataFrame(data=data)
    logging.info(f'{reports_path} - {filename}')
    if not os.path.isdir:
        os.mkdir(reports_path)
    df.to_csv(os.path.join(reports_path, filename), index=False)

def executeBenchmark(args):

    _, _, loader = get_zipped_dataloaders(args.data_root, args.batch_size, use_valid=True)
    label_to_classes = getLabelToClassMapping(os.path.join(os.getcwd(), args.data_root))

    for bench_type in args.bench_types:
        for arch, pol in args.arch_pol_tupl_ls:
            d = {'run': [], 'skip_n': [], 'bench_type': [], 'arch': [], 'pol': [], 'prec': [], 'rec': [], 'acc': [], 'f1': [], 'time': []}
            #config tqdm
            logging.info(f'Running {bench_type}-Bench on {arch}-{pol}...')

            skip_layers_list = args.skip_layers_values
            runs = args.runs

            if pol == 'none':
                arch_name = arch
                skip_layers_list = [0]
                runs = 1
            else:
                arch_name = f'{arch}-{pol}'
            with tqdm(total=(len(skip_layers_list) * runs), ncols=80, desc=f'Progress-{bench_type}-{arch}-{pol}') as pbar:
                for skip_n in skip_layers_list:
                    for run in range(runs):
                        prec = 0.0
                        rec = 0.0
                        acc = 0.0
                        f1 = 0.0
                        time = 0.0

                        try:
                            if bench_type == 'quality':
                                prec, acc, rec, f1 = executeQualityBench(arch_name, loader, skip_n, label_to_classes, args.batch_size)
                                #print(f'{run} - {skip_n} - {bench_type} - {arch} - {pol} - {prec:.6f} - {rec:.6f} - {acc:.6f} - {f1:.6f}')
                            elif bench_type == 'speed':
                                time = executeSpeedBench(arch_name, skip_n)
                                #print(f'{run} - {skip_n} - {bench_type} - {arch} - {pol} - {time:.6f}')
                            else:
                               print('Benchmark type not supported')
                               quit(1)
                        except Exception as e:
                            logging.info(f'Exception occured in {bench_type}: {e}\n continueing...')
                            print(f'run: {run}')
                            print(f'skip: {skip_n}')
                            print(f'bench_type: {bench_type}')
                            print(f'arch: {arch}')
                            print(f'pol: {pol}')
                            traceback.print_exc()
                            continue

                        d['run'].append(run)
                        d['skip_n'].append(skip_n)
                        d['bench_type'].append(bench_type)
                        d['arch'].append(arch)
                        d['pol'].append(pol)
                        d['prec'].append(prec)
                        d['rec'].append(rec)
                        d['acc'].append(acc)
                        d['f1'].append(f1)
                        d['time'].append(time)
                        
                        pbar.update(1)

            filename = f'{bench_type}-{arch}-{pol}-run.csv'
            storeReportToCSV(args.reports_path, filename, d)

if __name__ == "__main__":

    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    densenet_archs = ['densenet169']#['densenet121', 'densenet169']
    densenet_pol = ['-skip-last']#['none', '-skip', '-skip-last']
    densenet_archs = [x + y for x in densenet_archs for y in densenet_pol]
    densenet_archs = [x.replace('none', '') for x in densenet_archs]
    
    resnet_archs = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    resnet_pol = ['none', '-drop-rand-n', '-drop-last-rand-n']
    resnet_archs = [x + y for x in resnet_archs for y in resnet_pol]
    resnet_archs = [x.replace('none', '') for x in resnet_archs]
    
    args = parser.parse_args()

    architectures = densenet_archs + resnet_archs
    
    if 'only_arch' in args:
        if args.only_arch == 'resnet':
            architectures = resnet_archs
        elif args.only_arch == 'densenet':
            architectures = densenet_archs

    policies = list(set(densenet_pol + resnet_pol))

    arch_pol_tupl_ls = []
    for ls in architectures:
        if '-' in ls:
            fst, snd = ls.split('-', 1)
        else:
            fst = ls
            snd = 'none'
        arch_pol_tupl_ls.append((fst, snd))

    skip_layers_values = [1, 2, 3, 4, 8, 16, 32]
    bench_types = ['quality', 'speed']

    if args.bench_type is not None:
        if args.bench_type == 'quality':
            bench_types = ['quality']
        if args.bench_type == 'speed':
            bench_types = ['speed']

    runs = 10

    args.architectures = architectures
    args.policies = policies
    args.arch_pol_tupl_ls = arch_pol_tupl_ls
    args.skip_layers_values = skip_layers_values
    args.bench_types = bench_types
    args.runs = 30
    args.data_root = REDUCED_SET_PATH
    args.reports_path = os.path.join(os.getcwd(), 'reports')
    args.state_path = STATE_DICT_PATH
    args.batch_size = 1 if not 'batch_size' in args else args.batch_size
    logging.info(args)
    executeBenchmark(args)

