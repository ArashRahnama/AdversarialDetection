#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:55:59 2019

@author: Arash Rahnama
"""
from utility.trainer import train_classifier
from utility.trainer import train_detector
from utility.test import test_detector
from utility.test import test_detector_transfer
from utility.generate_adv_data import generate_adversarial_data
import os
###############################################################################
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
dataset_name = 'mnist'#'imagenet'
print('+'*20 + 'train_classifier' + '+'*20)
train_classifier(dataset_name=dataset_name)
###############################################################################
for attack in ['fgsm','pgdm']:
    for eps in [0.0,0.1,0.2,0.3,0.4]:
        print('+'*30, 'dataset:', dataset_name, ', attack:', attack, ', attack strength:', eps, '+'*30)
        generate_adversarial_data(dataset_name=dataset_name, attack=attack, eps=eps)
        print('+'*40 + ' Done! ' + '+'*40)
#################################################################################
for attack in ['fgsm','pgdm']:
    for eps in [0.1,0.2,0.3,0.4]:
        print('+'*40 + 'train_adversarial_detector' + '+'*40)
        train_detector(dataset_name=dataset_name, attack=attack, eps=eps, freeze=False)
        print('+'*40 + ' Done! ' + '+'*40)
#################################################################################
for attack in ['fgsm','pgdm']:
    for eps in [0.1,0.2,0.3,0.4]:
        print('+'*40 + 'test_adversarial_detector' + '+'*40)
        test_detector(dataset_name=dataset_name, attack=attack, eps=eps)
        print('+'*40 + ' Done! ' + '+'*40)   
#################################################################################
for dataset_name in ['mnist']:
    for target_dataset_name in ['mnist']:
        for attack in ['fgsm','pgdm']:
            for target_attack in ['fgsm','pgdm']:
                for eps in [0.1, 0.2, 0.3, 0.4]:
                    for target_eps in [0.1, 0.2, 0.3, 0.4]:
                        test_detector_transfer(dataset_name=dataset_name, attack=attack, eps=eps,
                                               target_dataset_name=target_dataset_name,
                                               target_attack=target_attack, target_eps=target_eps)
##################################################################################
