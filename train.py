import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
#from data.ImagenetDataset import get_imagenet_datasets

from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
from densenet import *
#from msdnet.models.msdnet import MSDNet

import os
import shutil
import time
import sys

from utils import get_msd_net_model, save_checkpoint, AverageMeter, get_batch_size_stats

# for repo:
DATA_PATH = "data/imagenet_images"
# for colab:
# DATA_PATH = "drive/My Drive/reducedAnytimeDnn/data/imagenet_images"
BATCH_SIZE = 1#8
NUM_WORKERS = 1#4

STAT_FREQUENCY = 1
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GPU_ID = None
START_EPOCH = 0
EPOCHS = 1
CHECKPOINT_INTERVALL = 10
CHECKPOINT_DIR = 'checkpoints'
ARCH = 'resnet50'

def main(argv):
    torch.cuda.empty_cache()

    n_gpus_per_node = torch.cuda.device_count()
    print(f"Found {n_gpus_per_node} GPU(-s)")

    # create model 
    model = ResNet.resnet50()
    
    
    if not torch.cuda.is_available():
      print("Using CPU for slow training process")
    else:
      print("Cuda is available")
      if GPU_ID is not None:
        torch.cuda.set_device(GPU_ID)
        model.cuda(GPU_ID)
      else:
        print("Using all available GPUs")
        model = nn.DataParallel(model).cuda()
    
    # loss function (criterion) and optimizer
    if torch.cuda.is_available():
      print("Move cross entropy to device")
      criterion = nn.CrossEntropyLoss().cuda()
    else:
      criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY)
    
    cudnn.benchmark = True

    train_loader, test_loader, _ = get_dataloaders_alt(
        DATA_PATH, 
        data="ImageNet", 
        use_valid=False, 
        save='save/default-{}'.format(time.time()),
        batch_size=BATCH_SIZE, 
        workers=NUM_WORKERS, 
        splits=['train', 'test'])
    
    # size of batch:
    print(get_batch_size_stats(train_loader))
    
    # train loop
    #for epoch in range(START_EPOCH, EPOCHS):
    best_acc = 0.0
    for epoch in range(START_EPOCH, EPOCHS):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        print('Running train loop')
        train(train_loader, model, criterion, optimizer, epoch)
        
        #evaluate the network on test set
        print('Compute accuracy')
        acc = validate(test_loader, model, criterion)
        
        # remember top acc
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        
        # safe model
        if epoch % CHECKPOINT_INTERVALL == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': ARCH,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, ARCH, CHECKPOINT_DIR)


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
        #print(res)
        return res

def adjust_learning_rate(optimizer, epoch):
    """
        Sets learning rate to default value, decayed by division with 10 every 25 epochs and 
        updates the lr in the optimizer.
    """
    lr = LEARNING_RATE * (0.1 ** (epoch // 25)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_load_time = AverageMeter('Data Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        if GPU_ID is not None:
            input = input.cuda(GPU_ID, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(GPU_ID, non_blocking=True)
        # time it takes to load data
        data_load_time.update(time.time() - end)
        
        # compute output of the current network
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        return

        # printing statistics every 2000 mini batch size
        if i % STAT_FREQUENCY == STAT_FREQUENCY - 1:
            print(f'Stats of Train loop {i} of {len(train_loader)}')
            # measure accuracy and record loss
            
            print(f'Epoch {epoch} - Iteration {i}/{len(train_loader)} - Loss {loss}')
            print(top1)
            print(top5)
            print(batch_time)
            print(data_load_time)
            return

def validate(val_loader, model, criterion):
    """Compute average accuracy, top 1 and top 5 accuracy"""
    model.eval()
    
    batch_time = AverageMeter('Batch Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i , (input, target) in enumerate(val_loader):
            # check if could be moved to cuda device
            if GPU_ID is not None:
                input = input.cuda(GPU_ID, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(GPU_ID, non_blocking=True)
                
            # compute output
            output = model(input)
            
            # compute loss
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(),input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            return top1.avg

            if i % STAT_FREQUENCY == STAT_FREQUENCY - 1:
                print(f'validation loop {i} of {len(val_loader)}')
                print(losses)
                print(top1)
                print(top5)
                print(batch_time)
                return top1.avg
    return top1.avg

if __name__ == "__main__":
   main(sys.argv)