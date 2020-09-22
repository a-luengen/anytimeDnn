#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.parallel
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim


from msdnet.dataloader import get_dataloaders_alt, get_dataloaders
from msdnet.args import arg_parser
from msdnet.adaptive_inference import dynamic_evaluate
import msdnet.models as models
from msdnet.op_counter import measure_model

import os
import shutil
import time
import datetime
import sys
import logging
import math

from utils import *

IS_DEBUG = True
DEBUG_ITERATIONS = 40

STAT_FREQUENCY = 20
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GPU_ID = None
START_EPOCH = 0
EPOCHS = 90
CHECKPOINT_INTERVALL = 30
CHECKPOINT_DIR = 'checkpoints'
ARCH = 'resnet101'

ARCH_NAMES = ['resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'msdnet']

# for repo:
#DATA_PATH = "data/imagenet_images"
# for colab:
DATA_PATH = "drive/My Drive/reducedAnytimeDnn/data/imagenet_images"
BATCH_SIZE = 18
NUM_WORKERS = 2

class Object(object):
  pass

args = None

try: 
  args = arg_parser.parse_args()
except:
  args = Object()
  args_dict = {
      'gpu': 'gpu:0',
      'use_valid': True,
      'data': 'ImageNet',
      'save': os.path.join(os.getcwd(), 'save'),
      'evalmode': None,
      'start_epoch': START_EPOCH,
      'epochs': EPOCHS,
      'arch': 'msdnet',
      'seed': 42,

      'grFactor': "1-2-4-4",
      'bnFactor': "1-2-4-4",
      'nBlocks': 3,
      'reduction': 0.5,
      'bottleneck': True,
      'prune': 'max',
      'growthRate': 16,
      'base': 4,
      'step': 4,
      'stepmode': 'even',
      
      'lr': LEARNING_RATE,
      'lr_type': 'multistep',
      'momentum': MOMENTUM,
      'weight_decay': WEIGHT_DECAY,
      'resume': False,
      'data_root': DATA_PATH,
      'batch_size': BATCH_SIZE,
      'workers': 4,
      'print_freq': STAT_FREQUENCY
  } 

  for key in args_dict:
    setattr(args, key, args_dict[key])
    #print(getattr(args, key))


if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 40

torch.manual_seed(args.seed)

def main():

    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    #model = getattr(models, args.arch)(args)
    #n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    #torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    #del(model)
        
    
    model = get_msd_net_model() #getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    elif torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    print("Print 2")
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print("Print 3")
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    print("Print 4")
    cudnn.benchmark = True

    temp = vars(args)
    for item in temp:
        print(item, ':', temp[item])
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print("Print 5")
    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return
    print("Print 1")
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            logging.info(f'Best var_prec1 {best_prec1}')

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    logging.info(f'Best val_prec1: {best_prec1:.4f} at epoch {best_epoch}')

    ### Test the final model

    logging.info('********** Final prediction results **********')
    validate(test_loader, model, criterion)

    return 

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        #input_var = torch.autograd.Variable(input)
        #target_var = torch.autograd.Variable(target)

        output = model(input)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logging.info(
                f'Epoch: [{epoch}][{i + 1}/{len(train_loader)}]\t'
                f'Time {batch_time.avg:.3f}\t'
                f'Data {data_time.avg:.3f}\t'
                f'Loss {losses.val:.4f}\t'
                f'Acc@1 {top1[-1].val:.4f}\t'
                f'Acc@5 {top5[-1].val:.4f}')

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda()

            #input_var = torch.autograd.Variable(input)
            #target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info(f'Epoch: [{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.avg:.3f}\t'
                      f'Data {data_time.avg:.3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Acc@1 {top1[-1].val:.4f}\t'
                      f'Acc@5 {top5[-1].val:.4f}')
    for j in range(args.nBlocks):
        logging.info(f' * prec@1 {top1[j].avg:.3f} prec@5 {top5[j].avg:.3f}')
    # logging.info(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
  
def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    logging.info("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    logging.info("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, args, is_best, filename, result):
    logging.info(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f'=> saving checkpoint {model_filename}')

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    logging.info(f"=> saved checkpoint '{model_filename}'")
    return


if __name__ == '__main__':
    curTime = datetime.datetime.now()
    #log_level = logging.INFO
    #if IS_DEBUG:
    log_level = logging.DEBUG

    #logging.basicConfig(filename=str(curTime) + ".log", level=log_level)
    logging.basicConfig(level=log_level)
    main()
