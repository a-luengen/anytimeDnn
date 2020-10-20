import os
import sys
import time
import logging
import datetime
import argparse
import traceback
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.backends.cudnn as cudnn

import msdnet.models

from utils import *
from data.ImagenetDataset import get_zipped_dataloaders, REDUCED_SET_PATH, FULL_SET_PATH

RUN_PATH = 'runs/'
DATA_PATH = REDUCED_SET_PATH
IS_DEBUG = True
DEBUG_ITERATIONS = 3
STAT_FREQUENCY = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GPU_ID = None
START_EPOCH = 0
EPOCHS = 2
CHECKPOINT_INTERVALL = 4 
CHECKPOINT_DIR = 'checkpoints'

LOG_FLOAT_PRECISION = ':6.4f'
BATCH_SIZE = 8

ARCH_NAMES = ['msdnet']

parser = argparse.ArgumentParser(description='Train several image classification network architectures.')
parser.add_argument('--arch', '-a', metavar='ARCH_NAME', type=str, default='msdnet', 
    choices=ARCH_NAMES, 
    help='Specify which kind of network architecture to train.')
parser.add_argument('--epoch', metavar='N', type=int, default=argparse.SUPPRESS, help='Resume training from the given epoch. 0-based from [0..n-1]')
parser.add_argument('--batch', metavar='N', type=int, default=argparse.SUPPRESS, help='Batchsize for training or validation run.')


def AddMSDNetArguments(args):
    growFactor = list(map(int, "1-2-4-4".split("-")))
    bnFactor = list(map(int, "1-2-4-4".split("-")))

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
        'test_interval': 10,

        'grFactor': growFactor,
        'bnFactor': bnFactor,
        'nBlocks': 5,
        'nChannels': 32,
        'nScales': len(growFactor),
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
        'workers': 1,
        'print_freq': STAT_FREQUENCY
    } 

    for key in args_dict:
        setattr(args, key, args_dict[key])
        #print(getattr(args, key))

def main(args):

    torch.cuda.empty_cache()

    n_gpus_per_node = torch.cuda.device_count()
    logging.info(f"Found {n_gpus_per_node} GPU(-s)")


    # MAIN LOOP
    #model = get_msd_net_model()
    model = msdnet.models.msdnet(args)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        logging.debug("Cuda is available.")
        logging.info("Using all available GPUs")
        for i in range(torch.cuda.device_count()):
            logging.info(f"gpu:{i} - {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model).cuda()
        logging.info("Moving criterion to device.")
        criterion = criterion.cuda()
        cudnn.benchmark = True
    else:
        logging.info("Using slow CPU training.")


    optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    calc_lr = lambda epoch: epoch // 30
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_lr)

    train_loader, val_loader, test_loader = get_zipped_dataloaders(args.data_root, args.batch_size, use_valid=True)

    best_prec1, best_epoch = 0.0, 0


    for epoch in range(EPOCHS):
        logging.info(f"Started Epoch{epoch + 1}/{EPOCHS}")
        # train()
        train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        # validate()
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)
        scheduler.step()

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            logging.info(f'Best val_prec1 {best_prec1}')
        
        if is_best or epoch % CHECKPOINT_INTERVALL == 0:
            save_checkpoint(getStateDict(model, epoch, 'msdnet', best_prec1, optimizer),
                            is_best, 
                            'msdnet', 
                            CHECKPOINT_DIR)

        if epoch % args.test_interval == 0:
            avg_loss, avg_top1, avg_top5 = validate(test_loader, model, criterion)


    logging.info(f'Best val_prec1: {best_prec1:.4f} at epoch {best_epoch}')

    logging.info('*************** Final prediction results ***************')
    validate(test_loader, model, criterion)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Batch Time', LOG_FLOAT_PRECISION)
    losses = AverageMeter('Loss', LOG_FLOAT_PRECISION)
    data_time = AverageMeter('Data Time', LOG_FLOAT_PRECISION)
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter(f'Top1-{i+1}', LOG_FLOAT_PRECISION))
        top5.append(AverageMeter(f'Top1-{i+1}', LOG_FLOAT_PRECISION))

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                img = img.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            data_time.update(time.time() - end)

            output = model(img)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target)

            losses.update(loss.item(), img.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1,5))
                top1[j].update(prec1.item(), img.size(0))
                top5[j].update(prec5.item(), img.size(0))

            batch_time.update(time.time() - end)
            
            if i % args.print_freq == 0:
                logging.info(f'Val - Epoch: [{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.avg:.3f}\t'
                      f'Data {data_time.avg:.3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Acc@1 {top1[-1].val:.4f}\t'
                      f'Acc@5 {top5[-1].val:.4f}')
            end = time.time()
            
            if IS_DEBUG and i == DEBUG_ITERATIONS:
                return losses.avg, top1[-1].avg, top5[-1].avg

    for j in range(args.nBlocks):
        logging.info(f'Validation: prec@1 {top1[j].avg:.3f} prec@5 {top5[j].avg:.3f}')
    logging.info(f'Final Validation: prec@1 {top1[-1].avg:.3f} prec@5 {top5[-1].avg:.3f}')
    
    return losses.avg, top1[-1].avg, top5[-1].avg

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter('Batch Time', LOG_FLOAT_PRECISION)
    data_time = AverageMeter('Data Time', LOG_FLOAT_PRECISION)
    losses = AverageMeter('Loss', LOG_FLOAT_PRECISION)
    top1, top5 = [],[]

    for i in range(args.nBlocks):
        top1.append(AverageMeter(f'Top1-{i+1}', LOG_FLOAT_PRECISION))
        top5.append(AverageMeter(f'Top5-{i+1}', LOG_FLOAT_PRECISION))
    
    model.train()
    end = time.time()

    running_lr = scheduler.get_last_lr()

    for i, data in enumerate(train_loader):
        
        data_time.update(time.time() - end)
        
        image, target = data

        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
                # time it takes to load data

        output = model(image)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target)
        
        losses.update(loss.item(), image.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), image.size(0))
            top5[j].update(prec5.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info(
                f'Train - Epoch: [{epoch}][{i + 1}/{len(train_loader)}]\t'
                f'Time {batch_time.avg:.3f}\t'
                f'Data {data_time.avg:.3f}\t'
                f'Loss {losses.val:.4f}\t'
                f'Acc@1 {top1[-1].val:.4f}\t'
                f'Acc@5 {top5[-1].val:.4f}')

        if IS_DEBUG and i == DEBUG_ITERATIONS:
            return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

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
    """Computes accuracy over the k top predictions for the values of k"""
    
    # reduce memory consumption on following calculations
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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



if __name__ == '__main__':
    args = parser.parse_args()
    
    AddMSDNetArguments(args)

    curTime = datetime.datetime.now()
    #log_level = logging.INFO
    #if IS_DEBUG:
    log_level = logging.DEBUG

    logging.basicConfig(filename=str(curTime) + ".log", level=log_level)
    #logging.basicConfig(level=log_level)

    try:
        main(args)
    except Exception as e:
        torch.cuda.empty_cache()
        print("Oh no! Bad things happened...")
        print(e)
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
