#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:55:18 2019

@author: Arash Rahnama
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

from models.detector import ResNet18
from data.load_data import Load_Data 
###############################################################################
def train_classifier(dataset_name='cifar10'):
    print(dataset_name)
    # create data and results directories, if it does not exist
    results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'
    data_dir = '/raid/dsroot/data/Theriac/data/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
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
    net.to(device)
    # initialize the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # train
    print('Training...')
    net.train()
    for epoch in range(num_epochs):
        print('epoch:',epoch)
        for data in dataloader_train:
            x = data[0].to(device)
            y = data[1].to(device)
            # zer the gradients
            optimizer.zero_grad()
            # forward/backward + optimize
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            print('loss:', loss.item(), end='\r')
        print('\nFnished epoch:', epoch)
    # save the final graph
    torch.save(net.state_dict(), results_dir+'classification_net_network_state_'+'final'+'.pt')
    # test
    print('Testing...')
    net.eval()
    preds_labels_all = []
    true_labels_all = []
    # iterate through test data
    for data in dataloader_test:
        x = data[0].to(device)
        y = data[1].to(device)
        preds_logits = net(x)
        preds_labels = np.argmax(preds_logits.cpu().detach().numpy(), axis=1)
        true_labels = y.cpu().detach().numpy()
        preds_labels_all.extend(preds_labels.tolist())
        true_labels_all.extend(true_labels.tolist())
    # compute accuracy
    print(classification_report(true_labels_all, preds_labels_all))
    # save predictions
    with open(results_dir+'classification_net_predictions_clean.json','w') as f:
        json.dump({'pred': preds_labels_all, 'true': true_labels_all}, f)
###############################################################################
def train_detector(dataset_name='cifar10', attack='fgsm', eps=0.1, freeze=False):
    print(dataset_name, attack, eps)
    # results directory
    results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+attack+'/'+str(eps)+'/'
    clean_set_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+'fgsm'+'/'+str(0.0)+'/'+'train/'
    # number of training batches
    num_batches = len([name for name in os.listdir(results_dir+'train/') if name[-3:] == '.pt'])
   
    if dataset_name == 'imagenet':
        num_classes = 1000
    else:
        num_classes = 10
        
    if freeze:
        net = ResNet18(num_classes=num_classes)
    else:
        # load ResNet18 graph where number of classes does not matter
        net = ResNet18(num_classes=1)
    net.defense_mode_on()
    print('Defense mode:', net.defense_mode)
    
    if freeze:
        ct=0
        for child in net.children():
            ct += 1
            if ct<11:
                print(child,'\n')
                for param in child.parameters():
                    param.requires_grad = False
    # is there a GPU?
    device_name = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print('Device:', device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)     
    net.to(device) 
    
    if freeze:   
    # initialize from trained graph
        model_results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'
        checkpoint_name = model_results_dir+'classification_net_network_state_'+'final'+'.pt'
        net.load_state_dict(torch.load(checkpoint_name))
    # initialize the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # train
    print('Training...')
    net.train()
    for epoch in range(100):
        for i in range(num_batches):
            x = torch.cat((torch.load(clean_set_dir+str(i)+'.pt'), 
                          torch.load(results_dir+'train/'+str(i)+'.pt')),
                          0).to(device)
            y = torch.cat((torch.zeros(int(x.size()[0]/2)), 
                           torch.ones(int(x.size()[0]/2))), 0).long().to(device)
            # zer the gradients
            optimizer.zero_grad()
            # forward/backward + optimize
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            print('loss:', loss.item(), end='\r')
        print('\nFnished epoch:', epoch)
    # save the final graph
    torch.save(net.state_dict(), results_dir+'detection_net_network_state_'+'final'+'.pt')
###############################################################################
