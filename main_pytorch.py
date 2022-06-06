#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:21:01 2021

@author: anonymous_ICML
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import copy

from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

import argparse

from models import *
from models_deepFC import *


plt.close('all')



# ask for the arguments
parser = argparse.ArgumentParser()

parser.add_argument('-en', '--exp_name',
                    type=str, default='exptorch', 
                    help="Experiment name")
parser.add_argument('-lt', '--learn_type',
                    type=str, default='ERIN', 
                    help="Learning rule: BP, ERIN, ERINsign, FA, DFA")
parser.add_argument('-r', '--n_runs',
                    type=int, default=1,
                    help="Number of simulations for each model")
parser.add_argument('-trep', '--train_epochs',
                    type=int, default= 100,
                    help="Number of training epochs")
parser.add_argument('-eta', '--eta',
                    type=float, default=0.01, 
                    help="Learning rate")
parser.add_argument('-do', '--dropout',
                    type=float, default=0.9, 
                    help="Dropout")
parser.add_argument('-Bstd', '--Bstd',
                    type=float, default=0.05, 
                    help="Std of fixed matrix")
parser.add_argument('-Bmz', '--B_mean_zero',
                    action='store_true', 
                    help="Choose if B matrix needs to have mean 0")
parser.add_argument('-check_cos_norm', '--check_cos_norm',
                    action='store_true', 
                    help="Compute antialignment angle and matrix norm during training")
parser.add_argument('-freeze_conv', '--freeze_conv',
                    action='store_true', 
                    help="Freeze convolutional layers for PEPITA")
parser.add_argument('-sqrt_conv', '--sqrt_conv',
                    action='store_true', 
                    help="Take the sqrt(n) for the update of convolutional layers with PEPITA")
parser.add_argument('-freeze_bn', '--freeze_bn',
                    action='store_true', 
                    help="Freeze the training of the batchnorm layers")
parser.add_argument('-eta_d', '--eta_decay',
                    action='store_true', 
                    help="If True, eta is decreased by a factor 0.1 every 60 epochs")
parser.add_argument('-decs', '--decay_scheme',
                    type=int, default=1,
                    help="Code for the learning rate decay scheme")
parser.add_argument('-is_pool', '--is_pool',
                    action='store_true', 
                    help="Choose if there is pooling in the network")
parser.add_argument('-is_fc', '--is_fc',
                    action='store_true', 
                    help="Choose if there  are only fc layers in the network")
parser.add_argument('-seed', '--seed',
                    default=None, 
                    help="Random seed. Set to None or to integer")
parser.add_argument('-ds', '--dataset',
                    type=str, default='cif', 
                    help="Dataset choice. Options: mn,cif,cif100,fmn,emn")
parser.add_argument('-ut', '--update_type',
                    type=str, default='mom', 
                    help="Update type: SGD, mom(entum), NAG, rmsprop, Adam ...")
parser.add_argument('-bs', '--batch_size',
                    default=50,type=int,
                    help="Batch size during training. Choose an integer")
parser.add_argument('-win', '--w_init',
                    type=str, default='he_uniform', #'he_uniform', 
                    help="Weight initialization type. Options: rnd, zero, ones, xav, he, he_uniform, nok, cir")
parser.add_argument('-mod', '--model',
                    type=str, default='Net1conv1fcXL_cif', #Net1conv1fcL
                    help="Network structure. Options NetFC1x1024DOcust,NetClark,NetGithub,NetGithub_cif,NetGithub_BP,NetGithub_cif_BP,NetConvHuge,NetConvHuge_BP,NetCroc_cif_BP,NetCroc_BP,NetCroc_cif_BP_bn,NetClark")
args = parser.parse_args()

#mnist = True

# save the arguments
# simulation set-up
exp_name = args.exp_name
n_runs = args.n_runs
train_epochs = args.train_epochs
eta = args.eta
print('Learning rate:',eta)
check_cos_norm = args.check_cos_norm
dropout = args.dropout
Bstd = args.Bstd
B_mean_zero = args.B_mean_zero
B_mean_zero = True
is_pool = args.is_pool
is_fc = args.is_fc
freeze_conv = args.freeze_conv
sqrt_conv = args.sqrt_conv
freeze_bn = args.freeze_bn
keep_rate = dropout
eta_decay = args.eta_decay
eta_decay = True # to be removed
decay_scheme = args.decay_scheme
seed = args.seed
dataset = args.dataset
w_init = args.w_init
# network set-up
learn_type = args.learn_type # current options are BP, ERIN
update_type = args.update_type # current options are SGD, mom(entum)
batch_size = args.batch_size
model = args.model
dataset = args.dataset

criterion = nn.CrossEntropyLoss()


# create folder to save all results
savepath = "res_"+exp_name+"_"+dataset+"_"+model+learn_type+"_"+update_type+"_"+str(batch_size)+"_"+w_init+"_"+"_rep"+str(n_runs)+"_tr"+str(train_epochs)

if eta_decay == True:
    savepath += "etad"+str(decay_scheme)
 
try:
    os.mkdir(savepath)
except OSError:
    print ("Creation of the directory %s failed" % savepath)
else:
    print ("Successfully created the directory %s " % savepath)
# prepare a file to write the results on     
filename = savepath+'/res_summary_'+exp_name+'.txt'
file = open(filename,'w')
file.write('Results for simulation with the following hyperparameters ')
file.write('\n Number of repetitions = ')
file.write(str(n_runs))
file.write('\n Training epochs = ')
file.write(str(train_epochs))
file.write('\n Learning rate = ')
file.write(str(eta))
file.write('\n Eta decay = ')
file.write(str(eta_decay))
file.write('\n F std = ')
file.write(str(Bstd))
file.write('\n Seed = ')
file.write(str(seed))
file.write('\n Dataset = ')
file.write(dataset)
file.write('\n Model = ')
file.write(model)
file.write('\n Learn type = ')
file.write(learn_type)
file.write('\n Batch size = ')
file.write(str(batch_size))
file.write('\n Update type = ')
file.write(update_type)
file.close()

    
# create variables to store results
train_acc_all = np.zeros((n_runs,train_epochs))
val_acc_all = np.zeros((n_runs,train_epochs))
test_acc_all = []


# load dataset
transform = transforms.Compose(
    [transforms.ToTensor()]) # this normalizes to [0,1]
if dataset == 'mn':
    ch_input = 1
    nout = 10
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    
elif dataset == 'cif':
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ch_input = 3
    nout = 10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    
elif dataset == 'cif100':
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    
    ch_input = 3
    nout = 100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)


    
    
# loop over the number of simulations
for r in range(n_runs):
    print('####### RUN {} #######'.format(r))
    if model == 'NetFC1x1024DOcust':
        net = NetFC1x1024DOcust(ch_input,nout)
    elif model == 'NetFC1x1024DOcust_cif':
        net = NetFC1x1024DOcust_cif(ch_input,nout)
    elif model == 'Net1conv1fcXL':
    	net = Net1conv1fcXL(ch_input,nout)
    elif model == 'Net1conv1fcXL_cif':
    	net = Net1conv1fcXL_cif(ch_input,nout)
        
    
    # set-up for BP
    if learn_type == 'BP':
        criterion = nn.CrossEntropyLoss()
        if update_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=eta)
        elif update_type == 'mom':
            optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)
        elif update_type == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=eta)
        
        if eta_decay:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # set-up for ERIN
    elif learn_type == 'ERIN':
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        for name, layer in net.named_modules():
            layer.register_forward_hook(get_activation(name))
        
        # define B 
        if dataset == 'mn':
            nin = 28*28*1
            nout = 10
        elif dataset == 'cif':
            nin = 32*32*3
            nout = 10
        elif dataset == 'cif100':
            nin = 32*32*3
            nout = 100
        sd = np.sqrt(6/nin)
        if B_mean_zero:
            B = (torch.rand(nin,nout)*2*sd-sd)*Bstd  # mean zero
        else:
            B = (torch.rand(nin,nout)*sd)*Bstd   # positive mean
        
        # save all weight shapes
        w_shapes = []
        for l_idx,w in enumerate(net.parameters()):
            if len(w.shape)>1:
                with torch.no_grad():
                    w_shapes.append(w.shape)
        # do one forward pass to get the activation size needed for setting up the dropout masks
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        #if is_fc:
        #    images = torch.flatten(images, 1) # flatten all dimensions except batch        
        outputs = net(images,do_masks=None)
        layers_act = []
        layers_key = []
        flag_fc = 0
        for key in activation:
            if 'fc' in key and 'bn' not in key or 'conv' in key and 'bn' not in key:
                layers_act.append(F.relu(activation[key]))
                layers_key.append(key)
                if flag_fc == 0 and 'fc' in key:
                    first_fc = len(layers_key)
                    flag_fc = 1
        # set up for momentum
        if update_type == 'mom':
            gamma = 0.9
            v_w_all = []
            for l_idx,w in enumerate(net.parameters()):
                if len(w.shape)>1:
                    with torch.no_grad():
                        v_w_all.append(torch.zeros(w.shape))     
                        
    # freeze the update of batchnorm layer if prescribed
    if freeze_bn:
        for name ,child in (net.named_children()):
            #if name.find('BatchNorm') != -1:
            if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d):
                for param in child.parameters():
                    param.requires_grad = False
                #print(name,'without grad')
            else:
                for param in child.parameters():
                    param.requires_grad = True 
                #print(name,'with grad')
    
    # load pretrained weights for convolutional layers and freeze the conv layers
    load_pretrained = False
    if load_pretrained:
        first_fc = 3
        for l_idx,w in enumerate(net.parameters()):
            if len(w.shape)>1 and l_idx+1 < first_fc:  # load only fc
            #if len(w.shape)>1:  # load both conv and fc
                with torch.no_grad():
                    w_np = np.loadtxt('NetGithub_w'+str(l_idx)+'.txt')
                    w_np = w_np.reshape(w.shape)
                    w += -w + w_np
        for name ,child in (net.named_children()):
            #if name.find('BatchNorm') != -1:
            if isinstance(child, nn.Conv2d):
                for param in child.parameters():
                    param.requires_grad = False
                #print(name,'without grad')
            else:
                for param in child.parameters():
                    param.requires_grad = True
                    
    # learning rate decay
    if eta_decay:
        decay_rate = 0.1
        if decay_scheme == 0:
            if dataset == 'mn':
                decay_epochs = [60]
            else:
                decay_epochs = [60,90]
        elif decay_scheme == 1:
            decay_epochs = [10,30,60]

    # train the model 
    test_accs = []
    losses = []
    for epoch in range(train_epochs):  # loop over the dataset multiple times
    
        # learning rate decay
        if eta_decay:
            if epoch in decay_epochs:
                if learn_type == 'BP':
                    scheduler.step()
                elif learn_type == 'ERIN':
                    eta = eta * decay_rate
                    print('At epoch {} learning rate decreased to {}'.format(epoch,eta))
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data
            
            if learn_type == 'BP':
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = net(inputs,do_masks=None)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            
            elif learn_type == 'ERIN':
                target_onehot = F.one_hot(target,num_classes=nout)
                # create dropout mask for the two forward passes
                do_masks = []
                for l_idx,l in enumerate(layers_act[:-1]):
                    if model == 'NetConvHuge' and l_idx < first_fc-1:
                        input1 = net.pool(l)
                    else:
                        input1 = l
                    do_mask = Variable(torch.bernoulli(input1.data.new(input1.data.size()).fill_(keep_rate)))/keep_rate
                    do_masks.append(do_mask)
             
                # forward pass 1 with original input --> keep track of activations
                outputs = net(inputs,do_masks)
                layers_act = []
                for key in activation:
                    if 'fc' in key and 'bn' not in key or 'conv' in key and 'bn' not in key:
                        layers_act.append(F.relu(activation[key]))
                
                error = outputs - target_onehot
                
                # modify the input with the error
                error_input = error @ B.T
                error_input = error_input.reshape_as(inputs)
                mod_inputs = inputs + error_input
                
                # forward pass 2 with modified input
                mod_outputs = net(mod_inputs,do_masks)
                mod_layers_act = []
                for key in activation:
                    if 'fc' in key and 'bn' not in key or 'conv' in key and 'bn' not in key:
                        mod_layers_act.append(F.relu(activation[key]))
                mod_error = mod_outputs - target_onehot
                
                # compute the delta_w for the batch
                delta_w_all = []      
                for l in range(len(layers_key)):
                    if 'fc' in layers_key[l] and 'bn' not in layers_key[l]:
                        #print('key for fc',layers_key[l],l)
                        if l == first_fc-1 and first_fc == len(layers_act): # only fc layers: case with only one fc layer after conv layers
                                                    
                            if is_pool == False:
                                delta_w = -error.T @ mod_layers_act[-2].flatten(1)
                            else:
                                delta_w = -error.T @ net.pool(mod_layers_act[-2]).flatten(1)
                        
                        
                        elif l == len(layers_act)-1: # last layer
                            #print('last fc')
                            if len(layers_act)>1:
                                delta_w = -mod_error.T @ mod_layers_act[-2]
                            else:
                                delta_w = -mod_error.T @ mod_inputs
        
                        elif l == first_fc-1: # first layer to be modified
                            #print('first fc --> apply pool, then reshape')
                            if first_fc > 1: # convolutional model    
                                if is_pool == False:
                                    input_to_fc = mod_layers_act[l-1]
                                else:
                                    input_to_fc = net.pool(mod_layers_act[l-1])
                            else: # fully connected model
                                input_to_fc = mod_inputs
                            
                            delta_w = -(layers_act[l] - mod_layers_act[l]).T @ input_to_fc.view(batch_size,-1)
        
                        elif l>first_fc-1 and l<len(layers_act)-1: # intermediate layers
                            #print('intermediate fc')
                            delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]
                            
                        delta_w_all.append(delta_w)
                        #print('delta_w',np.shape(delta_w))
                            
                    elif 'conv' in layers_key[l] and 'bn' not in layers_key[l]:
                        #print('key for conv',layers_key[l])
                        if l == 0:
                            inp = mod_inputs
                        else:
                            inp = mod_layers_act[l-1]
                            
                        if model == 'NetConvHuge':
                            inp = F.pad(inp,(1,1,1,1))
                        out_diff = layers_act[l] - mod_layers_act[l]
                        #print('out diff',np.shape(out_diff))
                        if freeze_conv == False:
                            if sqrt_conv == False:
                                delta_w = compute_delta_w_conv(inp,out_diff,w_shapes[l])
                            else:
                                delta_w = compute_delta_w_conv(inp,out_diff,w_shapes[l],sqrt=True)
                        else:
                            delta_w = torch.zeros(w_shapes[l])
                    
                        delta_w_all.append(delta_w)
                    
                # apply the weight change
                l_idx = 0
                for w in net.parameters():
                    if len(w.shape) > 1: # do not train the batchnorm layer
                        with torch.no_grad():
                            #print('w',w.shape,'dw',delta_w_all[l_idx].shape)
                            if update_type == 'SGD':
                                w += eta * delta_w_all[l_idx]/batch_size # specify for which layer
                            elif update_type == 'mom':
                                v_w_all[l_idx] = gamma * v_w_all[l_idx] + eta * delta_w_all[l_idx]/batch_size
                                w += v_w_all[l_idx]
    
                            l_idx += 1 # needed to skip batchnorm
                    
            
        
            # keep track of the loss
            loss = criterion(outputs, target)
            # print statistics
            running_loss += loss.item()
            
        curr_loss = running_loss / i
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / i))
        running_loss = 0.0
        losses.append(curr_loss)
            
            
        print('Testing...')
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for test_data in testloader:
                test_images, test_labels = test_data
                #test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
                # calculate outputs by running images through the network
                test_outputs = net(test_images,do_masks=None)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
    
        print('Test accuracy: {} %'.format(100 * correct / total))
        test_accs.append(100 * correct / total)
        
        # save the results for this network
        np.savetxt(savepath+'/losses_run'+str(r)+'.txt',losses)
        np.savetxt(savepath+'/test_acc_run'+str(r)+'.txt',test_accs)
    
    print('Finished Training') 
    
    
    
    
    



           
