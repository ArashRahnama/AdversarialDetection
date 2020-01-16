#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:35:48 2019

@author: Arash Rahnama
"""
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

from models.detector import ResNet18
from data.load_data import Load_Data 
from adversarial.attacks import fgsm, pgdm
###############################################################################
def generate_adversarial_data(dataset_name='cifar10', attack='fgsm', eps=0.1):
    print(dataset_name, attack, eps)
    results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+attack+'/'+str(eps)+'/'
    data_dir = '/raid/dsroot/data/Theriac/data'
    if not os.path.exists(results_dir+'train/'):
        os.makedirs(results_dir+'train/')
    if not os.path.exists(results_dir+'test/'):
        os.makedirs(results_dir+'test/')  
    dataloader_train, dataloader_test, num_classes, num_epochs = Load_Data(
                                                                 dataset_name, data_dir)
    # define model
    net = ResNet18(num_classes=num_classes)
    net.defense_mode_off()
    print('Defense mode:', net.defense_mode)
    # is there a GPU?
    device_name = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print('Device:', device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)     
    checkpoint_name = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+'classification_net_network_state_final.pt'
    net.load_state_dict(torch.load(checkpoint_name))
    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
#    net.eval()
    print('Generating adversarial data...')
    for i, data in enumerate(dataloader_train):
        x = data[0].to(device)
        y = data[1].to(device)
        if eps == 0.0:
            x = x
        else:
            if attack == 'fgsm':
                x = fgsm(net, x, y, criterion, eps=eps)
            elif attack == 'pgdm':
                x = pgdm(net, x, y, criterion, 0.00025, 100, eps)
        x = x.to(device)
        torch.save(x, results_dir+'train/'+str(i)+'.pt')
    preds_labels_all = []
    true_labels_all = []
    for i, data in enumerate(dataloader_test):
        x = data[0].to(device)
        y = data[1].to(device)
        if attack == 'fgsm':
            x = fgsm(net, x, y, criterion, eps=eps)
        elif attack == 'pgdm':
            x = pgdm(net, x, y, criterion, 0.00025, 100, eps)
        x = x.to(device)
        torch.save(x, results_dir+'test/'+str(i)+'.pt')
        preds_logits = net(x)
        preds_labels = np.argmax(preds_logits.cpu().detach().numpy(), axis=1)
        true_labels = y.cpu().detach().numpy()
        preds_labels_all.extend(preds_labels.tolist())
        true_labels_all.extend(true_labels.tolist())
    # compute accuracy
    print(classification_report(true_labels_all, preds_labels_all))
###############################################################################        
