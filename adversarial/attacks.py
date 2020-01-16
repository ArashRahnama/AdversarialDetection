#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:31:29 2019

@author: Arash Rahnama
"""
import torch
###############################################################################
# Fast Gradient Sign Method
def fgsm(net, x, y, loss_criterion, eps):
    # compute gradient for data
    x.requires_grad = True
    preds = net(x)
    net.zero_grad()
    loss = loss_criterion(preds, y)
    loss.backward()
    x_grad = x.grad
    # get the element-wise sign of gradient
    signed_x_grad = x_grad.sign()
    # create the adversarial input
    adv_x = x + (eps*signed_x_grad)
    return adv_x
###############################################################################    
# Projected Gradient Descent Method
def pgdm(net, x, y, loss_criterion, eps, steps, radius):
    # perturbations 
    pgd = x.new_zeros(x.shape)
    # create the adversarial input
    adv_x = x + pgd
    for step in range(steps):
        pgd = pgd.detach()
        x = x.detach()
        # compute gradient for data
        adv_x = adv_x.clone().detach()
        adv_x.requires_grad = True
        preds = net(adv_x)
        net.zero_grad()
        loss = loss_criterion(preds, y)
        loss.backward(create_graph=False, retain_graph=False)
        adv_x_grad = adv_x.grad
        # get the element-wise sign of gradient
        signed_adv_x_grad = adv_x_grad.sign()
        # create the adversarial input
        pgd = pgd + (eps*signed_adv_x_grad)
        pgd = torch.clamp(pgd, -radius, radius)
        adv_x = x + pgd
        return adv_x
###############################################################################
