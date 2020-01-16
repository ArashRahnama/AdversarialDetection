#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:38:27 2019

@author: Arash Rahnama
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

from models.detector import ResNet18
###############################################################################
def test_detector(dataset_name='cifar10', attack='fgsm', eps=0.1):
    print(dataset_name, attack, eps)
    results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+attack+'/'+str(eps)+'/'
    clean_set_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+'fgsm'+'/'+str(0.0)+'/'+'test/'
    #number of testing batches
    num_batches = len([name for name in os.listdir(results_dir+'test/') if name[-3:] == '.pt'])
    # load ResNet18 graph where number of classes does not matter
    net = ResNet18(num_classes=1)
    net.defense_mode_on()
    print('Defense mode:', net.defense_mode)
    # initialize from trained graph
    checkpoint_name = results_dir+'detection_net_network_state_'+'final'+'.pt'
    net.load_state_dict(torch.load(checkpoint_name))
    # is there a GPU?
    device_name = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print('Device:', device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)     
    net.to(device)
    # test
    print('Testing...')
    net.eval()
    preds_labels_all = []
    true_labels_all = []
    for i in range(num_batches):
        x = torch.cat((torch.load(clean_set_dir+str(i)+'.pt'), 
                      torch.load(results_dir+'test/'+str(i)+'.pt')),
                      0).to(device)
        y = torch.cat((torch.zeros(int(x.size()[0]/2)), 
                       torch.ones(int(x.size()[0]/2))), 0).long().to(device)
        preds_logits = net(x)
        preds_labels = np.argmax(preds_logits.cpu().detach().numpy(), axis=1)
        true_labels = y.cpu().detach().numpy()
        preds_labels_all.extend(preds_labels.tolist())
        true_labels_all.extend(true_labels.tolist())
    # compute accuracy
    print(classification_report(true_labels_all, preds_labels_all))
    with open(results_dir+'detection_net_predictions.json','w') as f:
        json.dump({'pred': preds_labels_all, 'true': true_labels_all}, f)
###############################################################################
def test_detector_transfer(dataset_name='cifar10', attack='fgsm', eps=0.1,
                          target_dataset_name='mnist', target_attack='fgsm', 
                          target_eps=0.1):
    print(dataset_name, attack, eps, '----->', target_dataset_name, 
          target_attack, target_eps)
    # directories of the original and target problems
    results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+attack+'/'+str(eps)+'/'
    target_results_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+target_dataset_name+'/'+target_attack+'/'+str(target_eps)+'/'
    clean_set_dir = '/raid/dsroot/data/Theriac/results/resnet18/'+dataset_name+'/'+'fgsm'+'/'+str(0.0)+'/'+'test/'
    # number of testing batches
    num_batches = len([name for name in os.listdir(target_results_dir+'test/') if name[-3:] == '.pt'])
    # load ResNet18 graph where number of classes does not matter
    net = ResNet18(num_classes=1)
    net.defense_mode_on()
    print('Defense mode:', net.defense_mode)
    # initialize from trained graph
    checkpoint_name = results_dir+'detection_net_network_state_'+'final'+'.pt'
    net.load_state_dict(torch.load(checkpoint_name))
    # is there a GPU?
    device_name = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print('Device:', device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)     
    net.to(device)
    # test
    print('Testing...')
    net.eval()
    preds_labels_all = []
    true_labels_all = []
    for i in range(num_batches):
        x = torch.cat((torch.load(clean_set_dir+str(i)+'.pt'), 
                      torch.load(target_results_dir+'test/'+str(i)+'.pt')),
                      0).to(device)
        y = torch.cat((torch.zeros(int(x.size()[0]/2)), 
                       torch.ones(int(x.size()[0]/2))), 0).long().to(device)
        preds_logits = net(x)
        preds_labels = np.argmax(preds_logits.cpu().detach().numpy(), axis=1)
        true_labels = y.cpu().detach().numpy()
        preds_labels_all.extend(preds_labels.tolist())
        true_labels_all.extend(true_labels.tolist())
    # compute accuracy
    print(classification_report(true_labels_all, preds_labels_all))
###############################################################################
