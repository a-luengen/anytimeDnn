import os
import torch
from sklearn import metrics
import logging
from timeit import default_timer as timer
import pandas as pn

from context import resnet, utils, data_loader, data_utils

BATCH_SIZE = 4
STATE_DICT_PATH = '../state'
DATA_PATH = '../data/imagenet_full'
STORE_RESULT_PATH = os.getcwd()


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

def with_timetracking(function):
    def inner(*args, **kwargs):
        start = timer()
        result = function(*args, **kwargs)
        end = timer()
        print(f'Took: {(end - start) * 1000:0.1f} [ms]')
        return result
    return inner

@with_timetracking
def executeQualityBench(arch_name: str, loader, skip_n: int, classes, batch_size: int):
    if skip_n <= 0:
        model = utils.getModel(arch_name)
    else:
        model = utils.getModelWithOptimized(arch_name, n=skip_n, batch_size=batch_size)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.eval()
    arch = arch_name.split('-')[0]
    model, _, _, _ = utils.resumeFromPath(os.path.join(STATE_DICT_PATH, f'{arch}_model_best.pth.tar'), model)
    grndT, pred = evaluateModel(model, loader, classes, batch_size)

    #logging.debug(metrics.classification_report(grndT, pred, digits=3))
    #prec = metrics.precision_score(grndT, pred, average='macro')
    #rec = metrics.recall_score(grndT, pred, average='macro')
    #f1 = metrics.f1_score(grndT, pred, average='macro')
    #acc = metrics.accuracy_score(grndT, pred)
    return grndT, pred

if __name__ == "__main__":
    experiment_data = list()

    logging.basicConfig(level=logging.INFO)

    model_name = 'resnet50-drop-rand-n'

    classes = data_utils.getLabelToClassMapping(DATA_PATH)
    _, _, loader = data_loader.get_zipped_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, use_valid=True)
    name = model_name.split('-')[0]
    logging.info(f'Execute Non-Skip Run for {name}')
    groundT, pred = executeQualityBench(name, loader, skip_n=0, classes=classes, batch_size=BATCH_SIZE)
    experiment_data.append({
        'skip_n': 0,
        'ground_truth': groundT,
        'prediction': pred,
        'configuration': [],
        'acc': metrics.accuracy_score(groundT, pred)
    })

    logging.info('Finished Non-Skip run. Switching to Skip-Run')

    for skip_layers in range(1, 7):
        for run in range(0, 30):
            logging.info(f'Executing Run {run + 1} of 30')
            groundT, pred = executeQualityBench(model_name, loader, skip_n=skip_layers, classes=classes, batch_size=BATCH_SIZE)
            policy = resnet.DropPolicies.getDropPolicy()
            layer_config = []
            if isinstance(policy, resnet.DropPolicies.ResNetDropRandNPolicy):
                layer_config = policy.getSkipConfigurationList()
            experiment_data.append({
                'skip_n': skip_layers,
                'ground_truth': groundT, 
                'prediction': pred, 
                'configuration': layer_config,
                'acc': metrics.accuracy_score(groundT, pred)
                })
            break

    df = pn.DataFrame(experiment_data)
    logging.info(f'Storing result at {STORE_RESULT_PATH}')
    df.to_csv(os.path.join(STORE_RESULT_PATH, 'resnet50_experiment_results.csv'), index=False)