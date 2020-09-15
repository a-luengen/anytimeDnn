import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import logging


def get_dataloaders(args):
    return get_dataloaders_alt(
        args.data_root, 
        args.data, 
        args.use_valid, 
        args.save, 
        args.batch_size, 
        args.workers,
        args.splits
        )


def get_dataloaders_alt(data_root, data, use_valid, save, batch_size, workers, splits):
    train_loader, val_loader, test_loader = None, None, None
    if data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    else:
        # ImageNet
        logging.info(f"Creating train_set and val_set loaders: Batch = {batch_size}, Worker = {workers}, Splits = {splits} ")
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

    if use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(save, 'index.pth')):
            logging.info('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(save, 'index.pth'))
        else:
            logging.info('!!!!!! Save train_set_index !!!!!!')
            os.mkdir(save)
            torch.save(train_set_index, os.path.join(save, 'index.pth'))

        if data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            #num_sample_valid = 50000
            # take 10% of the train set as validation set
            num_sample_valid = int(len(train_set) * 0.1)

        if 'train' in splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=workers, pin_memory=True)
        if 'val' in splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=workers, pin_memory=True)
        if 'test' in splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
    else:
        if 'train' in splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True)
        if 'val' or 'test' in splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
