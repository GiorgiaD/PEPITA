# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:02:06 2020

@author: anonymous_ICML
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *
from scipy import spatial
from numpy import linalg as LA


# layer class
class general_layer():
    def __init__(self,input_size,output_size,act,d_act,w_init):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.d_act = d_act
        self.w_accumulator = []
        
        if w_init == 'rnd':
            print('Random weight initialization')
            self.w = np.random.rand(output_size,input_size)/ \
                        np.sqrt(input_size)
        elif w_init == 'zero':
            print('Zero weight initialization')
            self.w = np.zeros((output_size,input_size))
        elif w_init == 'ones':
            print('Ones weight initialization')
            self.w = np.ones((output_size,input_size))
        elif w_init == 'xav':
            print('Xavier weight initialization (uniform)')
            nin = input_size; nout = output_size
            sd = np.sqrt(6.0 / (nin + nout))
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd, sd))
                    self.w[j,i] = x
        elif w_init == 'he':
            print('Kaiming He weight initialization (normal)')
            nin = input_size; nout = output_size
            sd = np.sqrt(2.0 / nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.normal(loc = mu, scale = sd))
                    self.w[j,i] = x
        elif w_init == 'he_uniform':
            print('Kaiming He weight initialization (uniform)')
            nin = input_size; nout = output_size
            limit = np.sqrt(6.0 / nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-limit, limit))
                    self.w[j,i] = x
        elif w_init == 'nok':
            print('Nokland weight initialization')
            nin = input_size; nout = output_size
            sd = 1.0 / np.sqrt(nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd,sd))
                    self.w[j,i] = x
        if w_init == 'cir':
            print('Ciresan weight initialization')
            nin = input_size; nout = output_size
            sd = 0.05
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd,sd))
                    self.w[j,i] = x  
        self.delta_w_batch = np.zeros_like(self.w)
        
        
    def forward(self,x,dropout,training,update_type,learn_type,compute_diff=False):
        if learn_type in ['ERIN','ERINsign'] and compute_diff==True: # condition satisfied except at first presentation step
            self.m_output = np.copy(self.output) # keeps in memory activity at the previous step
            self.m_x = np.copy(self.x)
            self.m_a = np.copy(self.a)
            #print("m = ",self.m)
        self.x = x
        
                
        # for all learning types
        if update_type != 'NAG':
            self.a = self.w @ x
            w = self.w
        elif update_type == 'NAG':
            self.a = self.w_lookahead @ x
        self.output = self.act(self.a)  # current output 
        
        # apply dropout FIX DROPOUT FOR ERIN
        if dropout != 1.0:
            if training:
                if learn_type != 'ERIN' and learn_type != 'ERINsign':
                    self.drop_mask = np.random.binomial(1,dropout,size = np.shape(self.a))/dropout
                elif learn_type in ['ERIN','ERINsign']:
                    if compute_diff == False: # first pass
                        self.drop_mask = np.random.binomial(1,dropout,size = np.shape(self.a))/dropout
                        self.memorized_drop_mask = self.drop_mask
                    elif compute_diff == True: # successive passes
                        self.drop_mask = self.memorized_drop_mask
            else:
                self.drop_mask = 1.
            self.output *= self.drop_mask
            
        if learn_type in ['ERIN','ERINsign'] and compute_diff==True:
            self.diff = self.m_output - self.output
            
        return self.output
        
# network class
class general_network():
    def __init__(self,layers_size,act_list,d_act_list,learn_type,batch_size,update_type,keep_variants,w_init,sample_passes,VERBOSE=False):
        self.layers_size = layers_size
        self.n_layers = len(self.layers_size)
        print(self.n_layers)
        self.learn_type = learn_type
        self.batch_size = batch_size
        self.update_type = update_type
        self.keep_variants = keep_variants
        self.sample_passes = sample_passes
        if self.sample_passes > 1:
            if self.learn_type not in ['ERIN','ERINsign']:
                self.effective_batch_size = self.batch_size * self.sample_passes
            else:
                self.effective_batch_size = self.batch_size * (self.sample_passes-1)
        else:
            self.effective_batch_size = self.batch_size
        self.layers = []
        for i in range(self.n_layers-1):
            new_layer = general_layer(self.layers_size[i],self.layers_size[i+1],act_list[i],d_act_list[i],w_init)
            self.layers.append(new_layer)
            
        # if needed in the model, initialize the F matrix and the activity storage vectors
        if self.learn_type in ['ERIN','ERINsign']:
            # F matrix --> propagates the error to the input
            self.layers[-1].F = np.zeros((self.layers[0].input_size,self.layers_size[-1]))
            if w_init == 'nok':
                print('Nokland weight initialization for ERIN')
                nin = self.layers[0].input_size
                nout = self.layers_size[-1]
                sd = 1.0 / np.sqrt(nout)
                mu = 0.0
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-sd, sd))
                        self.layers[-1].F[i,j] = x
            elif w_init == 'ones':
                print('Ones weight initialization for ERIN')
                self.layers[-1].F = np.ones((self.layers[0].input_size,self.layers_size[-1]))
            
            elif w_init == 'he_uniform':
                print('Kaiming He weight initialization for ERIN')
                nin = self.layers[0].input_size
                nout = self.layers_size[-1]
                limit = np.sqrt(6.0 / nin)
                mu = 0.0
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-limit, limit))
                        self.layers[-1].F[i,j] = x         
            else:
                print('Nokland weight initialization for ERIN')
                nin = self.layers[-1].input_size
                nout = self.layers_size[-1]
                sd = 1.0 / np.sqrt(nout)
                mu = 0.0
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-sd, sd))
                        self.layers[-1].F[i,j] = x
                        
            self.layers[-1].F = self.layers[-1].F * 0.05
            # further reduce the sdt if using only the sign of the error
            if self.learn_type in ['ERINsign']:
                self.layers[-1].F = self.layers[-1].F * 0.1
                        
            
        elif self.learn_type in ['FA']:
            for idx in range(len(self.layers)-1,0,-1):
                if w_init == 'nok':
                    print('Nokland weight initialization for FA')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                elif w_init == 'he_uniform':
                    print('Kaiming He weight initialization (uniform) for feedback weights')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    limit = np.sqrt(6.0 / nin)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-limit, limit))
                            self.layers[idx].B[i,j] = x  
                else:
                    print('Nokland weight initialization for FA')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                            
                        
            #self.layers[idx].B = np.random.rand(self.layers[idx].input_size,self.layers[idx].output_size)    
        elif self.learn_type in ['DFA']:
            for idx in range(len(self.layers)-1,0,-1):
                if w_init == 'nok':
                    print('Nokland weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                elif w_init == 'he_uniform':
                    print('Kaiming He weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    limit = np.sqrt(6.0 / nin)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-limit, limit))
                            self.layers[idx].B[i,j] = x         
                else:
                    print('Nokland weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
            
            
        # initialize variables according to the optimizer
        print('Update type is:'+self.update_type)
        if self.update_type == 'SGD':
            pass
        elif self.update_type =='mom':
            print('setting up parameters for momentum optimizer')
            self.gamma = 0.9
            for l in self.layers:
                l.v_w = np.zeros((l.output_size,l.input_size))
        elif self.update_type == 'NAG':
            print('setting up parameters for NAG optimizer')
            self.gamma = 0.9
            for l in self.layers:
                l.v_w = np.zeros((l.output_size,l.input_size))
                l.w_lookahead = np.copy(l.w)
        elif self.update_type =='rmsprop':
            print('setting up parameters for rmsprop optimizer')
            for l in self.layers:
                l.sqr_grad = np.zeros_like(l.w)
        elif self.update_type =='Adam':
            print('setting up parameters for Adam optimizer')
            for l in self.layers:
                #initialize l.vel, l.sqr, l.t
                l.vel = np.zeros_like(l.w)
                l.sqr = np.zeros_like(l.w)
                l.timestep = 1
        
    def forward(self,x,target,dropout,training,error_input=None,compute_diff=False):
        self.target=target
        if self.learn_type in ['ERIN','ERINsign'] and compute_diff==True:
            x += error_input # incorporates modulated error at previous time step into the input, except at first presentation step
            
        for l in self.layers[:-1]:
            x = l.forward(x,dropout,training,self.update_type,self.learn_type,compute_diff)
        x = self.layers[-1].forward(x,dropout=1.0,training=training,update_type=self.update_type,learn_type=self.learn_type,compute_diff=compute_diff)
        self.output = x
        self.error = self.output-target
        return self.output,self.error

    def learning(self,error,eta,dropout):
        # compute delta_a (depends on the method)
        self.layers[-1].delta_a = error
        for idx in range(len(self.layers)-2,-1,-1):
            if self.learn_type=='BP':
                if self.update_type != 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx+1].w.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                elif self.update_type == 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx+1].w_lookahead.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                
                    
            elif self.learn_type=='FA':
                self.layers[idx].delta_a = (self.layers[idx+1].B @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                
                
            elif self.learn_type=='DFA':
                self.layers[idx].delta_a = (self.layers[idx+1].B @ error) * self.layers[idx].d_act(self.layers[idx].a)
                
        
            elif self.learn_type in ['ERIN','ERINsign']:
                self.layers[idx].delta_a = self.layers[idx].diff # this also works on test (check, seems too good to be true)
                    
        if dropout != 1.0:
            for idx in range(len(self.layers)-2,-1,-1):
                self.layers[idx].delta_a *= self.layers[idx].drop_mask     
        

        # weight_update (depends on the chosen optimization method)
        for idx_l,l in enumerate(self.layers):
            if self.update_type == 'SGD':
                l.delta_w_batch += -l.delta_a * l.x.T
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    l.w += eta * l.delta_w_batch / self.effective_batch_size
                    l.delta_w_batch = np.zeros_like(l.w)
                else:
                    self.new_batch = False
                    
            elif self.update_type == 'mom':
                # step always performed
                l.delta_w_batch += l.delta_a * l.x.T 
                # step performed only at the end of the minibatch
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    l.v_w = self.gamma * l.v_w + eta * l.delta_w_batch / self.effective_batch_size
                    l.w += -l.v_w
                    l.delta_w_batch = np.zeros_like(l.w)
                    
                else:
                    self.new_batch = False
                   
            
        
    def train(self,x_list,target_list,x_list_test,target_list_test,train_epochs,sample_passes,eta,dropout,shuffling,eta_decay,deformation,test_as_val,zeromean,plots,validation,savepath,r,check_cos_norm):
        # set validation set differently depending on distortion or not
        if validation:
            if deformation == False and test_as_val == False:
                x_list, x_val, target_list, target_val = train_test_split(x_list, target_list, test_size=0.1, random_state=1)    
            else:
                print('Use full testing set for validation')
                x_val = x_list_test
                target_val = target_list_test
            val_size = len(x_val)
            val_pred_all = []    
            val_targs = []
        self.val_acc_all = []
        dataset_size = len(x_list)
        self.acc_all = []
        self.dropout = dropout
        E_curve = []
        pred_all = []
        targs = []
        points = 20
        if deformation:
            x_list_old = x_list
            target_list_old = target_list
            
        # check cosine similarity before training AND matrix norm
        #check_cos_norm = True
        if self.learn_type == 'ERIN' and check_cos_norm:
            angles = []
            w_all = []
            norm_w0 = []
            for l_idx,l in enumerate(self.layers):
                w_all.append(np.copy(l.w))
                if l_idx == 0:
                    norm_w0.append(LA.norm(l.w))
                    print('norm of w at layer {} is {}'.format(l_idx,norm_w0))
            w_prod = w_all[0].T
            for idx in range(1,len(w_all)):
                w_prod = w_prod @ w_all[idx].T
                print(np.shape(w_prod))
            w_prod = w_prod.flatten()
            B_flat = np.array(self.layers[-1].F).flatten()
            cos = 1-spatial.distance.cosine(w_prod,B_flat)
            arccos = np.arccos(cos)*180/np.pi
            print('Angle between Ws and B',arccos)
            angles.append(arccos)
       
        # perform the training loop
        for e in range(train_epochs):
            # shuffle the training set
            if shuffling==True and deformation==False:
                x_list, target_list = shuffle(x_list, target_list, random_state = 0)
            # apply deformation
            if deformation:
                x_list, target_list = distortion(x_list_old,target_list_old)
                # normalize to the interval [-1,1] if zeromean is True
                if zeromean:
                    #print("before: min = {} , max = {}".format(np.min(x_list),np.max(x_list)))
                    for i in range(len(x_list)):
                        x_list[i] = x_list[i]*2 - 1
            if train_epochs>9:
                if e%int(train_epochs/10)==0:
                    print('Training epoch {}/{}'.format(e,train_epochs))
            else:
                print('Training epoch {}/{}'.format(e,train_epochs))
            if eta_decay:
                if e in [60,90]:
                    eta = np.max((eta*(0.1),1e-6))
                    print("Learning rate at epoch {} decreased to {}".format(e,eta))
            acc = []
            val_accuracy = []
            self.new_batch = True
            
            for s in range(dataset_size): 
                #print("**********  sample {}  **********".format(s))
                self.s = s
                x = x_list[s]
                target = target_list[s]
                self.target = target
                for p in range(sample_passes): 
                    #print("Pass number",p)
                    targs.append(np.argmax(target))
                    
                    if self.learn_type != 'ERIN' and self.learn_type != 'ERINsign':
                        y,self.error = self.forward(x,target,dropout,training=True)
                        self.learning(self.error,eta,dropout)
                    
                    elif self.learn_type == 'ERIN':
                        if p == 0: # no learning, only compute output
                            y,self.error = self.forward(x,target,dropout,training=True,
                                                        error_input=None,compute_diff=False)
                        if p > 0: # apply the weight change only after the first pass
                            error_input = self.layers[-1].F @ self.error
                            x = np.copy(x_list[s])
                            y,self.error = self.forward(x,target,dropout,training=True,
                                                        error_input=error_input,compute_diff=True)
                            self.learning(self.error,eta,dropout)
                            
                            
                    elif self.learn_type == 'ERINsign':
                        if p == 0: # no learning, only compute output
                            y,self.error = self.forward(x,target,dropout,training=True,
                                                        error_input=None,compute_diff=False)
                        if p > 0: # apply the weight change only after the first pass
                            error_input = self.layers[-1].F @ np.sign(self.error)
                            x = np.copy(x_list[s])
                            y,self.error = self.forward(x,target,dropout,training=True,
                                                        error_input=error_input,compute_diff=True)
                            self.learning(self.error,eta,dropout)
                            
                    # save the error
                    E_curve.append(np.sum(abs(self.error)))
                    self.pred = onehotenc(np.argmax(y),np.size(y))
                    pred_all.append(np.argmax(self.pred))
                    if np.argmax(y) == np.argmax(target):
                        acc.append(1)
                    else:
                        acc.append(0)
                
                #print("target={}, pred={} at {}%".format(np.argmax(target),np.argmax(y),np.round(np.max(y),2)))
                    
            # check cosine similarity during training AND matrix norm
            if self.learn_type == 'ERIN' and check_cos_norm:
                w_all = []
                for l_idx,l in enumerate(self.layers):
                    w_all.append(np.copy(l.w))
                    if l_idx == 0:
                        norm_w0.append(LA.norm(l.w))
                        #print('norm of w at layer {} is {}'.format(l_idx,norm_w0))
                w_prod = w_all[0].T
                for idx in range(1,len(w_all)):
                    w_prod = w_prod @ w_all[idx].T
                    #print(np.shape(w_prod))
                w_prod = w_prod.flatten()
                B_flat = np.array(self.layers[-1].F).flatten()
                cos = 1-spatial.distance.cosine(w_prod,B_flat)
                arccos = np.arccos(cos)*180/np.pi
                #print('Angle between Ws and B',arccos)
                angles.append(arccos)
                    
            self.acc_all.append(np.mean(acc))
            if train_epochs>9:
                if e%int(train_epochs/10)==0:
                    print('Training accuracy = {}'.format(self.acc_all[-1]))
            else:
                print('Training accuracy = {}'.format(self.acc_all[-1]))
             
            if e%1 == 0:    
                np.savetxt(savepath+'/train_acc_tot.txt',np.array([self.acc_all]))
                if check_cos_norm:
                    np.savetxt(savepath+'/angles.txt',np.array([angles]))
                    np.savetxt(savepath+'/Anorm.txt',np.array([norm_w0]))
                
            # save the weights
            for i in range(self.n_layers-1):
                #np.savetxt(savepath+'/weights_layer'+str(i)+'.txt',self.layers[i].w)
                pass
            # perform validation
            if validation:
                for s in range(val_size):                           
                    x = x_val[s]
                    target = target_val[s]
                    self.target = target
                    y,self.error = self.forward(x,target,dropout,training=False)
                    # save the error
                    self.pred = onehotenc(np.argmax(y),np.size(y))
                    #print('target {} pred {}'.format(np.argmax(target),np.argmax(self.pred)))
                    val_pred_all.append(np.argmax(self.pred))
                    if np.argmax(y) == np.argmax(target):
                        val_accuracy.append(1)
                    else:
                        val_accuracy.append(0)
                    val_targs.append(np.argmax(target))
                
                self.val_acc_all.append(np.mean(val_accuracy))
                if train_epochs>9:
                    if e%int(train_epochs/10)==0:
                        print('Validation accuracy = {}'.format(self.val_acc_all[-1]))
                else:
                    print('Validation accuracy = {}'.format(self.val_acc_all[-1]))
                if e%1 == 0: 
                    np.savetxt(savepath+'/val_acc_tot.txt',np.array([self.val_acc_all]))
                
            if plots:    
                plt.figure()
                plt.plot(val_pred_all,'*',label='Prediction')
                plt.plot(val_targs,'.',label='Target')
                plt.title(str(self.learn_type)+' - Validation')
                plt.legend() 
        
        
        if plots:
            plt.figure()
            plt.plot(pred_all,'*',label='Prediction')
            plt.plot(targs,'.',label='Target')
            plt.title(str(self.learn_type))
            plt.legend() 
            
        
        return E_curve, self.acc_all, self.val_acc_all
            
    
    def test(self,x_list,target_list,plots,plots_test=False):
        dataset_size = len(x_list)
        pred_all = []    
        targs = []
        accuracy = []
                
        for s in range(dataset_size):   
            if dataset_size>9:
                if s%int(dataset_size/10)==0:
                    #print('Testing sample {}/{}'.format(s,dataset_size))
                    pass
            else:
                #print('Testing sample {}/{}'.format(s,dataset_size))
                pass
                
            x = x_list[s]
            target = target_list[s]
            self.target = target
            y,self.error = self.forward(x,target,self.dropout,training=False)
            # save the error
            self.pred = onehotenc(np.argmax(y),np.size(y))
            #print('target {} pred {}'.format(np.argmax(target),np.argmax(self.pred)))
            pred_all.append(np.argmax(self.pred))
            if np.argmax(y) == np.argmax(target):
                accuracy.append(1)
            else:
                accuracy.append(0)
            targs.append(np.argmax(target))
        
        accuracy_mean = np.mean(accuracy)
        
        if plots or plots_test:    
            plt.figure()
            plt.plot(pred_all,'*',label='Prediction')
            plt.plot(targs,'.',label='Target')
            plt.title(str(self.learn_type))
            plt.legend() 
        
        return accuracy_mean

