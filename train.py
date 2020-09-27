import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from msdnet.dataloader import get_dataloaders_alt
from resnet import ResNet
import densenet.densenet as dn

from data.ImagenetDataset import get_zipped_dataloaders

import os
import shutil
import time
import datetime
import sys
import logging

from utils import *

################################- Constants for Checkpoints -###################################
# File name containing checkpoint for given architecture: <arch_name>_<EPOCH>_checkpoint.pth.tar
LAST_CHECKPOINT_EPOCH = 0
# True: Resume from a checkpoint file stored in the checkpoint subdirectory 
# or use default if none is found
# False: Do not resume from any possible checkpoint file
RESUME = True
ARCH = 'resnet101'
ARCH_NAMES = ['resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169']
################################################################################################

IS_DEBUG = False
DEBUG_ITERATIONS = 40
STAT_FREQUENCY = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GPU_ID = None
START_EPOCH = 0
EPOCHS = 90
CHECKPOINT_INTERVALL = 4 
CHECKPOINT_DIR = 'checkpoints'

# for repo:
# raw images
# DATA_PATH = "data/imagenet_images"
# zipped preprocessed images
DATA_PATH = "data/imagenet_full"
# for colab:
# DATA_PATH = "drive/My Drive/reducedAnytimeDnn/data/imagenet_images"
BATCH_SIZE = 4
NUM_WORKERS = 1

def main(argv):
    torch.cuda.empty_cache()

    n_gpus_per_node = torch.cuda.device_count()
    logging.info(f"Found {n_gpus_per_node} GPU(-s)")

    # create model 
    model = getModel(ARCH)

    logging.info(f"Training Arch:{ARCH}")

    if not torch.cuda.is_available():
      logging.warning("Using CPU for slow training process")
    else:
      logging.debug("Cuda is available")
      if GPU_ID is not None:
        logging.info(f"Using specific GPU: {GPU_ID}")
        logging.warning("This will reduce the training speed significantly.")
        torch.cuda.set_device(GPU_ID)
        model.cuda(GPU_ID)
      else:
        logging.info("Using all available GPUs")
        for i in range(torch.cuda.device_count()):
            logging.info(f"gpu:{i} - {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model).cuda()
    
    # loss function (criterion) and optimizer
    if torch.cuda.is_available():
      logging.info("Move cross entropy to device")
      criterion = nn.CrossEntropyLoss().cuda()
    else:
      criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY)
    
    cudnn.benchmark = True
    
    train_loader, test_loader, _ = get_zipped_dataloaders(
        os.path.join(os.getcwd(), "data", "imagenet_full"), 
        BATCH_SIZE, 
        use_valid=True)


    # size of batch:
    logging.debug(get_batch_size_stats(train_loader))
    

    if RESUME:
        model, optimizer, start_epoch, best_acc  = resumeFromPath(
            os.path.join(os.getcwd(), CHECKPOINT_DIR, ARCH + f"_{LAST_CHECKPOINT_EPOCH}_" + CHECKPOINT_POSTFIX), 
            model, 
            optimizer)
    else:
        start_epoch = START_EPOCH
        best_acc = 0.0
    
    checkpoint_time = AverageMeter('Checkpoint Time', ':6.3f')
    epoch_time = AverageMeter('Epoch Time', ':6.3f')
    # train loop
    end = time.time()
    for epoch in range(start_epoch, EPOCHS):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        logging.debug('Running train loop')
        train(train_loader, model, criterion, optimizer, epoch)
        
        #evaluate the network on test set
        logging.debug('Compute accuracy')
        acc = validate(test_loader, model, criterion)
        
        # remember top acc
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        
        # safe model
        if epoch % CHECKPOINT_INTERVALL == 0 or is_best or IS_DEBUG:
            start = time.time()
            save_checkpoint(
                getStateDict(
                    model, epoch, 
                    ARCH, best_acc, 
                    optimizer), 
                is_best, ARCH, os.path.join(os.getcwd(), CHECKPOINT_DIR))
            checkpoint_time.update(time.time() - start)
            logging.info(checkpoint_time)
        if IS_DEBUG:
            break
        epoch_time.update(time.time() - end)
        end = time.time()
        logging.info(epoch)
        logging.info(f"Avg-Epoch={epoch_time.avg}sec, Avg-Checkp.={checkpoint_time.avg}sec")
    logging.info(f"Best accuracy: {best_acc}")

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

def adjust_learning_rate(optimizer, epoch):
    """
        Sets learning rate to default value, decayed by division with 10 every 25 epochs and 
        updates the lr in the optimizer.
    """
    if not epoch % 25 == 0 and epoch > 0:
        return
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
    for i, (img, target) in enumerate(train_loader):
        
        if GPU_ID is not None:
            img = img.cuda(GPU_ID, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(GPU_ID, non_blocking=True)
        # time it takes to load data
        data_load_time.update(time.time() - end)
        
        # compute output of the current network
        output = model(img)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # printing statistics every 2000 mini batch size
        if i % STAT_FREQUENCY == STAT_FREQUENCY - 1:            
            logging.info(f'Epoch {epoch} Train loop - Iteration {i}/{len(train_loader)} - Loss {loss}')
            logging.info(top1)
            logging.info(top5)
            logging.info(batch_time)
            logging.info(data_load_time)
        if IS_DEBUG and i == DEBUG_ITERATIONS:
                break
    logging.info(f"Epoch {epoch} train summary: Avg. Acc@1={top1.avg:6.2f} - " 
        + f"Avg. Acc@5={top5.avg:6.2f} - " 
        + f"Avg. Batch={batch_time.avg:6.2f}sec - "
        + f"Avg. DataLoad={data_load_time.avg}sec" )

def validate(val_loader, model, criterion):
    """Compute average accuracy, top 1 and top 5 accuracy"""
    model.eval()
    
    batch_time = AverageMeter('Batch Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i , (img, target) in enumerate(val_loader):
            # check if could be moved to cuda device
            if GPU_ID is not None:
                img = img.cuda(GPU_ID, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(GPU_ID, non_blocking=True)
                
            # compute output
            output = model(img)
            
            # compute loss
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(),img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)


            if i % STAT_FREQUENCY == STAT_FREQUENCY - 1:
                logging.info(f'validation loop {i} of {len(val_loader)}')
                logging.info(losses)
                logging.info(top1)
                logging.info(top5)
                logging.info(batch_time)
            if IS_DEBUG and i == DEBUG_ITERATIONS:
                return top1.avg
    return top1.avg

def loadAndEvaluate():
    model = getModel(ARCH)

    if os.path.exists(os.path.join(CHECKPOINT_DIR, ARCH + '_model_best.pth.tar')):
        logging.debug("Loading best model")
        load_path = os.path.join(CHECKPOINT_DIR, ARCH + '_model_best.pth.tar')
    else:
        logging.debug("Loading default model")
        load_path = os.path.join(CHECKPOINT_DIR, ARCH + '_checkpoint.pth.tar')
    
    logging.debug('Loading: ' + load_path)

    model, _, _ = resumeFromPath(load_path, model)

    logging.debug('Loading Test Data..')

    _, _, testLoader = get_zipped_dataloaders(DATA_PATH, BATCH_SIZE, use_valid=True)
    grndT, pred = evaluateModel(model, testLoader)

    printStats(grndT, pred)

def evaluateModel(model, loader):
    model.eval()

    with torch.no_grad():
        logging.debug(f'Loaded testData with {len(loader.dataset)} testImages and {BATCH_SIZE} images per batch.')

        classes = getClasses(os.path.join(DATA_PATH, 'val'))
        grndT, pred = [], []
        for i, (images, labels) in enumerate(loader):
            logging.debug(f"Evaluating: {i}-th iteration")
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pred = pred + [classes[predicted[k]] for k in range(BATCH_SIZE)]
            grndT = grndT + [classes[labels[j]] for j in range(BATCH_SIZE)]
            
            if IS_DEBUG and i == DEBUG_ITERATIONS:
                break
        return grndT, pred

if __name__ == "__main__":
    curTime = datetime.datetime.now()

    log_level = logging.INFO
    if IS_DEBUG:
        log_level = logging.DEBUG

    logging.basicConfig(filename=str(curTime) + ".log", level=log_level)
    main(sys.argv)
    logging.info(f"Top1 Accuracy: {loadAndEvaluate()}")
