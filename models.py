#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:24:59 2021

@author: anonymous_ICML
"""

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

# convolutional models for PEPITA 
# note that the softmax operation needs to be removed to use the models with BP

class Net1conv1fcXL(nn.Module):
    def __init__(self,ch_input,nout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5,bias=False)
        self.fc1 = nn.Linear(4608, nout,bias=False)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, do_masks):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.softmax(self.fc1(x))
        return x
    
class Net1conv1fcXL_cif(nn.Module):
    def __init__(self,ch_input,nout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,bias=False)
        self.fc1 = nn.Linear(6272, nout,bias=False)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, do_masks):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.softmax(self.fc1(x))
        return x
    


# for conv models
def compute_delta_w_conv(inp,out_diff,w_shape,stride=1,sqrt=False, plot=False, plot2d=False):
    delta_w = torch.zeros(w_shape)
    ch_out = w_shape[0] # number of output channels
    size_out = out_diff.shape[-1] # size of output map
    ch_in = w_shape[1] # number of input channels
    size_in = inp.shape[-1] # size of input map
    ks = w_shape[2] # kernel height and width
    bs = out_diff.shape[0]
    cnt = 0
    #print(ch_out,size_out,ch_in,size_in,ks,bs)
    if plot:
        fig, axs = plt.subplots(1, size_out**2, figsize=(12, 3), sharey=False)
        fig2, axs2 = plt.subplots(1, size_out**2, figsize=(12, 3), sharey=False)
    if plot2d:
        figb, axsb = plt.subplots(1, size_out**2, figsize=(12, 3), sharey=False)
        fig2b, axs2b = plt.subplots(1, size_out**2, figsize=(12, 3), sharey=False)
    for r in range(0,size_out): # loop over all the output rows
        for c in range(0,size_out): # loop over all the output columns
            #print(ch_out,size_out,ch_in,size_in,ks)
            #print("r,c",r,c)
            inp_r_start = stride*r
            inp_r_end = stride*r+ks
            inp_c_start = stride*c
            inp_c_end = stride*c+ks
            this_out_diff = out_diff[:,:,r,c]
            this_inp = inp[:,:,inp_r_start:inp_r_end,inp_c_start:inp_c_end]
            partial = ev(this_out_diff, this_inp, bs, ch_in, ch_out, ks).reshape_as(delta_w) # gives the right answer
            delta_w += partial
            if plot:
                axs[cnt].imshow(partial.detach().numpy()[0,0])
                axs2[cnt].imshow(delta_w.detach().numpy()[0,0])
                if plot2d:
                    axsb[cnt].imshow(partial.detach().numpy()[1,0])
                    axs2b[cnt].imshow(delta_w.detach().numpy()[1,0])
            cnt += 1
    if sqrt == False:
        delta_w *= 1./cnt          # dw = dw/n
    else:
        delta_w *= 1./np.sqrt(cnt)
    return delta_w


def ev(this_out_diff, this_inp, bs, chin, chout, ks):
    prod_mul = torch.mul(this_out_diff.reshape(bs,chout,1,1,1), this_inp.reshape(bs,1,chin,ks,ks))
    prod_mul = torch.mean(prod_mul,axis=0)  # average across batchsize
    return prod_mul


    


