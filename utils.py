import argparse
import os
import shutil
import torch
from msdnet.models.msdnet import MSDNet

def get_msd_net_model():
    grFact = '1-2-4-4'
    bnFact = '1-2-4-4'
    obj = argparse.Namespace()
    obj.nBlocks = 1
    obj.nChannels = 32
    obj.base = 4
    obj.stepmode = 'even'
    obj.step = 4
    obj.growthRate = 16
    obj.grFactor = list(map(int, grFact.split('-')))
    obj.prune = 'max'
    obj.bnFactor = list(map(int, bnFact.split('-')))
    obj.bottleneck = True
    obj.data = 'ImageNet'
    obj.nScales = len(obj.grFactor)
    obj.reduction = 0.5 # compression of densenet
    return MSDNet(obj).train()

def save_checkpoint(state, is_best: bool, arch: str, checkpoint_dir: str, filename=None):
    
    if filename is None:
        filename = f'{arch}_checkpoint.pth.tar'

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    torch.save(state, filename)
    print(checkpoint_dir, filename)
    shutil.move(filename, os.path.basename(checkpoint_dir))
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, f'{arch}_model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_batch_size_stats(loader):
    txt = ""
    for i, (input, target) in enumerate(loader):
      if i == 0:
        element_size_in_byte = input.element_size()
        n_elements = input.nelement()
        size_in_byte = element_size_in_byte * n_elements
        txt += f"Input:\n{n_elements} Elements times {element_size_in_byte} bytes is {size_in_byte}\n"
        element_size_in_byte = target.element_size()
        n_elements = target.nelement()
        size_in_byte = element_size_in_byte * n_elements
        txt += f"Target:\n{n_elements} Elements times {element_size_in_byte} bytes is {size_in_byte}"
        break
    return txt