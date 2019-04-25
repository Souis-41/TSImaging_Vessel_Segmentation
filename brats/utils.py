#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:58:59 2019

@author: sure
"""
import math
from functools import partial
import pdb
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping


K.set_image_dim_ordering('th')

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def Resize(self, img, target_slice,interpolation): ##(?,288,288) -->  (16,288,288)
    
    pdb.set_trace()
    
    channel, width, height = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    scale_z = float(channel/target_slice)
    scaled_img = np.zeros((target_slice, width, height))
    
    if interpolation == "linear":
        
        for i in range(target_slice):
            ori_index_float = i * scale_z
            floor = np.floor(ori_index_float)
            ceil = np.ceil(ori_index_float)
            if ceil > target_slice:
                ceil = int(target_slice - 1)
            scaled_img[i] = img[ceil] * (ori_index_float - floor) + img[floor] * (ceil - ori_index_float)
        
    elif interpolation == "nearest":
        for i in range(target_slice):
            ori_index_float = i * scale_z
            around = int(np.around(ori_index_float))
            if around > target_slice - 1:
                around = target_slice - 1
            scaled_img[i] = img[around]
        
    return scaled_img  ##(16,288,288)