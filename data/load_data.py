#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:48:52 2019

@author: Arash Rahnama
"""
import os
import torchvision
from torch.utils.data import DataLoader

def Load_Data(dataset_name='cifar10', data_dir = '/raid/dsroot/data/'):
    # data transform
    if dataset_name == 'mnist':
        transfs = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.Grayscale(3),
                torchvision.transforms.ToTensor()
                ])
    elif dataset_name == 'imagenet':
        transfs = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()])
    else:
        transfs = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor()
                ])   
    # select the dataset
    if dataset_name == 'cifar10':
        num_classes = 10
        num_epochs = 100
        dataset_train = torchvision.datasets.CIFAR10(data_dir,
                                                     train=True, transform=transfs,
                                                     target_transform=None, download=True)
        dataset_test = torchvision.datasets.CIFAR10(data_dir,
                                                     train=False, transform=transfs,
                                                     target_transform=None, download=True)
    elif dataset_name == 'mnist':
        num_classes = 10
        num_epochs = 35
        dataset_train = torchvision.datasets.MNIST(data_dir,
                                                   train=True, transform=transfs,
                                                   target_transform=None, download=True)
        dataset_test = torchvision.datasets.MNIST(data_dir,
                                                  train=False, transform=transfs,
                                                  target_transform=None, download=True)
    elif dataset_name == 'svhn':
        num_classes = 10
        num_epochs = 100
        dataset_train = torchvision.datasets.SVHN(data_dir,
                                                  train=True, transform=transfs,
                                                  target_transform=None, download=True)
        dataset_test = torchvision.datasets.SVHN(data_dir,
                                                 train=False, transform=transfs,
                                                 target_transform=None, download=True)
    elif dataset_name == 'imagenet':
        num_classes = 1000
        num_epochs = 100
        data_dir = '/raid/dsroot/data/ILSVRC2012'
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        dataset_train = torchvision.datasets.ImageFolder(train_dir, 
                                                         transform=transfs)
        dataset_test = torchvision.datasets.ImageFolder(test_dir, 
                                                        transform=transfs)
        
    # set up data loaders
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True,
                                  pin_memory=True, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False,
                                 pin_memory=True, num_workers=4)
    return dataloader_train, dataloader_test, num_classes, num_epochs
