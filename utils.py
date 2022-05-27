# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:02:06 2020

@author: GiorgiaDellaferrera
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# activation functions
def sigm(x):
    return 1/(1+np.exp(-x))
def d_sigm(x):
    return sigm(x) * (1-sigm(x))
def relu(x):
    return np.maximum(x,0)
def step_f(x,bias=0):
    return np.heaviside(x,bias)
def Lrelu(x,leakage=0.1):
    output = np.copy(x)
    output[output<0] *= leakage
    return output
def d_Lrelu(x,leakage=0.1):
    return np.clip(x>0,leakage,1.0)
def d_step_f(x):
    return 1-np.square(np.tanh(x))
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1-np.square(tanh(x))
def tanh_ciresan(x):
    A = 1.7159
    B = 0.6666
    return np.tanh(B*x)*A
def d_tanh_ciresan(x):
    A = 1.7159
    B = 0.6666
    return A*B*(1-np.square(tanh(B*x)))
def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
def onehotenc(idx,size):
    arr = np.zeros((size,1))
    arr[idx] = 1
    return arr
# to compute alignment of matrix
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1 = 2.*(v1 - np.min(v1))/np.ptp(v1)-1
    v2 = 2.*(v2 - np.min(v2))/np.ptp(v2)-1
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

# prepare dataset
def dataset_simple(n_input,n_output,n_samples,seed=None,VERBOSE=False,plots=False):
    x_list = []
    target_list = []
    for s in range(n_samples):
        # generate the sample and target
        if s == 0:
            x = np.zeros((n_input,1))
            #np.random.seed(633)
            idx_l = np.random.choice(n_input,2,replace=False)
            for i_l in idx_l:
                x[i_l] = 1
        if VERBOSE:
            print('s',s)    
        if s>0:
            flag = 1
            while flag>0:
                if VERBOSE:
                    print('drawing ')
                x = np.zeros((n_input,1))  
                idx_l = np.random.choice(n_input,2,replace=False)
                for i_l in idx_l:
                    x[i_l] = 1
                if VERBOSE:
                    print('check')
                flag=0
                if VERBOSE:
                    print('fl',flag)
                for i in range(len(x_list)):
                    if VERBOSE:
                        print(i)
                    if np.array_equal(x,x_list[i]):
                        flag+=1
                        if VERBOSE:
                            print('update: flag = ',flag)
                if VERBOSE:
                    print('final ',flag)        
                #plt.figure()
                #plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
                #plt.title('attempt')
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))

        target = np.zeros((n_output,1))
        #idx = np.random.choice(np.arange(n_output))
        idx = s
        target[idx] = 1
        # add it to the dataset
        x_list.append(x)
        target_list.append(target)
    return x_list, target_list


def dataset_mnist(n_samples,seed=None,plots=False):
    print('Loading mnist')
    x_max = 255
    x_min = 0
    # import mnist
    import keras
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        #i=0 # to be removed
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using digits:")
        print(idx_list)
    else:
        print("using the full mnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    # train
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    # test
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_emnist(n_samples,seed=None,plots=False):
    print('Loading emnist')
    x_max = 255
    x_min = 0
    # import emnist balanced
    from extra_keras_datasets import emnist
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using letters:")
        print(idx_list)
    else:
        print("using the full emnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 47
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_fmnist(n_samples,seed=None,plots=False):
    print('Loading fmnist')
    x_max = 255
    x_min = 0
    # import mnist
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using clothes:")
        print(idx_list)
    else:
        print("using the full fmnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_cifar(n_samples,seed=None,plots=False):
    print('Loading cifar10')
    x_max = 255
    x_min = 0
    # import cifar10
    import keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    class_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using classes:")
        print(idx_list)
        print("corresponding to labels:")
        for id_ in idx_list:
            print(class_labels[id_[0]])
    else:
        print("using the full cifar10 dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure(figsize=(2,2))
            plt.imshow(x.reshape((int(np.sqrt(n_input/3)),int(np.sqrt(n_input/3)),3)))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_cifar100(n_samples,seed=None,plots=False):
    print('Loading cifar100')
    x_max = 255
    x_min = 0
    # import cifar10
    import keras
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    #class_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using classes:")
        print(idx_list)
        print("corresponding to labels:")
        #for id_ in idx_list:
            #print(class_labels[id_[0]])
    else:
        print("using the full cifar100 dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 100
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure(figsize=(2,2))
            plt.imshow(x.reshape((int(np.sqrt(n_input/3)),int(np.sqrt(n_input/3)),3)))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test


def dataset_debug(n_samples,seed=None,plots=False):
    print('Loading dataset for debugging')
    x_list =[np.array([[1.,0.,0.,0.]]).T]
    target_list = [np.array([[0,0.5]]).T]
    x_list_test = x_list
    target_list_test = target_list
        
    return x_list, target_list, x_list_test, target_list_test

def distortion(x_train, y_train):
    #from keras.preprocessing.image import ImageDataGenerator
    from image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        elastic_RGB=[34,6],
        fill_mode='constant',
        vertical_flip=False,
        horizontal_flip=False)
    #print('before',np.shape(x_train))
    #print('before mean',np.mean(x_train))
    x_train = np.reshape(x_train,(np.shape(x_train) + (1,)))
    x_train = np.reshape(x_train,(np.shape(x_train)[0],int(np.sqrt(np.shape(x_train)[1])),int(np.sqrt(np.shape(x_train)[1])),1))
    #print('input shape',np.shape(x_train))
    datagen.fit(x_train)
    batches = 0
    for x_deformed, y_deformed in datagen.flow(x_train, y_train, batch_size=60000):
        batches += 1
        if batches >= 1:
            break
    #x_plot = np.reshape(x_deformed,(np.shape(x_deformed)[0],np.shape(x_deformed)[1],np.shape(x_deformed)[2]))
    #plt.figure()
    #plt.imshow(x_plot[0,:,:])
    #plt.colorbar()
    x_deformed = np.reshape(x_deformed,(np.shape(x_deformed)[0],np.shape(x_deformed)[1]*np.shape(x_deformed)[2],1))
    #print('after',np.shape(x_deformed))
    #print('after mean',np.mean(x_deformed))
    x_max = np.max(x_deformed)
    x_min = np.min(x_deformed)
    x_deformed = (x_deformed-x_min)/(x_max-x_min)
    #print('after normaliz',np.mean(x_deformed))
    return x_deformed, y_deformed

