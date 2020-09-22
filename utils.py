import argparse
import os
import shutil
import logging
import torch
from sklearn import metrics
from msdnet.models.msdnet import MSDNet
from resnet import ResNet
import densenet.densenet as dn

def get_msd_net_model():
    grFact = '1-2-4-4'
    bnFact = '1-2-4-4'
    obj = argparse.Namespace()
    obj.nBlocks = 1
    obj.nChannels = 224 # For CIFAR 32
    obj.base = 4
    obj.stepmode = 'even'
    obj.step = 4
    obj.growthRate = 16
    obj.grFactor = list(map(int, grFact.split('-')))
    obj.prune = 'max'
    obj.bnFactor = list(map(int, bnFact.split('-')))
    obj.bottleneck = True
    obj.data = 'ImageNet'
    obj.nScales = len(obj.grFactor) # 4 Scales
    obj.reduction = 0.5 # compression of densenet
    return MSDNet(obj)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt
        self.max = 0
        self.min = float('inf')

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.max < val: self.max = val
        if self.min > val: self.min = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '}) ({min' + self.fmt + '}) ({max' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_batch_size_stats(loader):
    txt = ""
    for i, (images, target) in enumerate(loader):
        if i == 0:
            element_size_in_byte = images.element_size()
            n_elements = images.nelement()
            size_in_byte = element_size_in_byte * n_elements
            txt += f"Input:\n{n_elements} Elements times {element_size_in_byte} bytes is {size_in_byte}\n"
            element_size_in_byte = target.element_size()
            n_elements = target.nelement()
            size_in_byte = element_size_in_byte * n_elements
            txt += f"Target:\n{n_elements} Elements times {element_size_in_byte} bytes is {size_in_byte}"
            return txt

def printStats(ground_truth, predicted):
    logging.info("Confussion matrix:")
    logging.info(metrics.confusion_matrix(ground_truth, predicted))
    logging.info("Recall and precision:")
    logging.info(metrics.classification_report(ground_truth, predicted, digits=3))

def getClasses(data_path: str):
    class_list = os.listdir(data_path)
    logging.debug(class_list)
    logging.debug(len(class_list))
    return class_list

def getModel(arch: str):
    logging.info(f"Loading model: {arch}")
    model = None
    if arch == 'resnet18':
        model = ResNet.resnet18()
    elif arch == 'resnet34':
        model = ResNet.resnet34()
    elif arch == 'resnet50':
        model = ResNet.resnet50()
    elif arch == 'resnet50-pol':
        model = ResNet.resnet50(use_policy=True)
    elif arch == 'resnet101':
        model = ResNet.resnet101()
    elif arch == 'resnet152':
        model = ResNet.resnet152()
    elif arch == 'densenet':
        model = dn.DenseNet3(3, 40)
    elif arch == 'densenet121':
        model = dn.DenseNet4([6, 12, 24, 16], 40, growth_rate=32)
    elif arch == 'densenet169':
        model = dn.DenseNet4([6, 12, 32, 32], 40, growth_rate=32)
    elif arch == 'msdnet':
        model = get_msd_net_model()
    else:
        model = ResNet.resnet50()
    return model

def getStateDict(model, epoch : int, arch : str, best_acc: float, optimizer):
    return {
        'epoch': epoch,
        'arch' : arch,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

CHECKPOINT_POSTFIX = '_checkpoint.pth.tar'
CHECKPOINT_BEST_POSTFIX = '_model_best.pth.tar'

def save_checkpoint(state, is_best: bool, arch: str, checkpoint_dir: str, filename=None):
    if filename is None:
        filename = state["arch"] + CHECKPOINT_POSTFIX

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    torch.save(state, filename)
    logging.debug(os.path.join(checkpoint_dir, filename))

    target = os.path.join(os.path.basename(checkpoint_dir), filename)
    if os.path.exists(target):
        os.remove(target)
    shutil.move(filename, os.path.basename(checkpoint_dir))

    if is_best:
        best_filename = state["arch"] + CHECKPOINT_BEST_POSTFIX
        best_filePath = os.path.join(checkpoint_dir, best_filename)
        source = os.path.join(checkpoint_dir, filename)
        shutil.copyfile(source, best_filePath)

def resumeFromPath(path : str, model, optimizer):
    start_epoch = 0
    best_prec1 = 0.0

    if not os.path.isfile(path):
        print(f'No file found {path}')
        logging.info(f"=> no checkpoint found at '{path}'")
        return model, start_epoch, best_prec1

    logging.debug(f"=> loading checkpoint {path}")
    
    if not (torch.cuda.is_available() and isinstance(model, torch.nn.DataParallel)):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # remove 'module.' string before keys in state_dict
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            name = k.replace('module.', '') # remove 'module.'
            new_state_dict[name] = v
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_acc']
    logging.info(f"=> loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")

    return model, optimizer, start_epoch, best_prec1