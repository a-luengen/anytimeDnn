import logging
import torch
import argparse
import os
from sklearn import metrics
from timeit import default_timer as timer
from utils import getModel, resumeFromPath
from data.ImagenetDataset import get_zipped_dataloaders, REDUCED_SET_PATH
from data.utils import getLabelToClassMapping
from typing import Tuple, List

from benchmark import storeReportToCSV

parser = argparse.ArgumentParser(description='Benchmark MSDNet variants.')
parser.add_argument('--batch_size', metavar='N', type=int, default=1, help='Batchsize for training or validation run.')
parser.add_argument('--bench_type', type=str, default=None, choices=['quality', 'speed'], help='Execute only the specfied benchmark type.')
parser.add_argument('--runs', metavar='N', type=int, default=30, help='Number of runs to collect data for each item to benchmark.')
parser.add_argument('--data_root', type=str, default=REDUCED_SET_PATH, help='Root path for a prepared zipped dataset.')
parser.add_argument('--report_path', type=str, default='reports', help='Root path for storing reports.')
parser.add_argument('--state_path', type=str, default=os.path.join(os.getcwd(), 'state'), help='Absolute path to the directory containing checkpoints for the model.')

def runSpeedBench(args, arch: str, max_classifications: int) ->float:

    model = getModel(arch)
    model.setMaxClassifiers(max_classifications)

    tensor = torch.rand(1, 3, 224, 224)

    # warmup
    temp_res = model(tensor)

    start = timer()
    temp_res = model(tensor)
    end = timer()

    return (end - start) * 1000

def executeSpeedBench(args, model_max: Tuple[str, int]):

    for arch, max_pred in model_max:
        measurements = {'classifier': [], 'arch': [], 'time': []}
        for max_classifications in range(1, max_pred + 1):

            for _ in range(args.runs):
                arch_name = f'{arch}{max_pred}'
                result = runSpeedBench(args, arch_name, max_classifications)

                measurements['classifier'].append(max_classifications)
                measurements['arch'].append(arch_name)
                measurements['time'].append(result)

        storeReportToCSV(
            os.path.join(os.getcwd(), args.report_path), 
            f'speed-{arch_name}-none-run.csv', 
            measurements)


def evaluateModel(args, model, loader, classes) -> (List[float], List[float]):
    pred, grndT = [], []
    for (images, labels) in iter(loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
        outputs = model(images)

        if isinstance(outputs, List):
            outputs = outputs[-1]

        _, predicted = torch.max(outputs, 1)
        pred = pred + [classes[predicted[k]] for k in range(min(args.batch_size, labels.shape[0]))]
        grndT = grndT + [classes[labels[j]] for j in range(min(args.batch_size, labels.shape[0]))]
    return grndT, pred


def getDataLoader(args):
    _, _, loader = get_zipped_dataloaders(args.data_root, args.batch_size, use_valid=True)
    return loader

def getClassificationValues(predT, grndT) -> (float, float, float, float):
    return (
            metrics.accuracy_score(grndT, predT),
            metrics.precision_score(grndT, predT, average='macro'),
            metrics.recall_score(grndT, predT, average='macro'),
            metrics.f1_score(grndT, predT, average='macro')
        )

def runQualityBench(args, arch_name: str, max_classification: int, loader) -> (float, float, float, float):
    loader = getDataLoader(args)
    label_to_classes = getLabelToClassMapping(os.path.join(os.getcwd(), args.data_root))

    model = getModel(arch_name)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model.setMaxClassifiers(max_classification)
    model, _ , _, _ = resumeFromPath(os.path.join(args.state_path, f'{arch_name}_model_best.pth.tar'), model)
    model.eval()

    predT, grndT = evaluateModel(args, model, loader, label_to_classes)

    return getClassificationValues(predT, grndT)

def executeQualityBench(args, model_max: Tuple[str, int]):
    loader = getDataLoader(args)

    for arch, max_classifications in model_max:
        stats = {'classifier': [], 'arch': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
        for max_cls in range(1, max_classifications + 1):
            arch_name = f'{arch}{max_classifications}'

            acc, prec, rec, f1 = runQualityBench(args, arch_name, max_cls, loader)

            stats['classifier'].append(max_cls)
            stats['arch'].append(arch_name)
            stats['acc'].append(acc)
            stats['prec'].append(prec)
            stats['rec'].append(rec)
            stats['f1'].append(f1)

        storeReportToCSV(args.report_path, f'quality-{arch_name}-run.csv', stats)

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    model_and_max = [('msdnet', 4), ('msdnet', 5), ('msdnet', 10)]
    model_and_max = [('msdnet', 10)]
    print(args)
    print(model_and_max)

    if args.bench_type is None or args.bench_type == 'speed':
        executeSpeedBench(args, model_and_max)

    if args.bench_type is None or args.bench_type == 'quality':
        executeQualityBench(args, model_and_max)