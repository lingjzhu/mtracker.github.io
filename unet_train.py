#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:53:49 2018

@author: luke
"""

import numpy as np
import cnn_model as cnn
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard
import os



path_to_train_x = './data/train_x'
path_to_train_y = './data/train_y_4'
path_to_validation_x = './data/validation_x'
path_to_validation_y = './data/validation_y_4'



batch_size = 32
input_size = [128,128,3]
epochs = 50
train_step = round(15822/32)+1
validation_step = round(1758/32)



def normalize(x):
    return (x-np.mean(x))/np.std(x)


def trim_scale(x,threshold=100):
    x = np.where(x<threshold,0,x)
    return x/np.max(x)


for loss in ['dice','class_xentropy',"compound"]:
    train_x_datagen_no_aug = dict(preprocessing_function=normalize)
    train_y_datagen_no_aug = dict(preprocessing_function=trim_scale)
    
    
    train_x_datagen = ImageDataGenerator(**train_x_datagen_no_aug)
    train_y_datagen = ImageDataGenerator(**train_y_datagen_no_aug)
    
        
    seed = 233
    train_x_generator = train_x_datagen.flow_from_directory(
        path_to_train_x,
        target_size=(input_size[0], input_size[1]),
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        seed=seed)
    
    train_y_generator = train_y_datagen.flow_from_directory(
        path_to_train_y,
        target_size=(input_size[0], input_size[1]),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)
    
    
    validation_x_datagen_no_aug = dict(preprocessing_function=normalize)
    validation_y_datagen_no_aug = dict(preprocessing_function=trim_scale)
    
    
    validation_x_datagen = ImageDataGenerator(**validation_x_datagen_no_aug)
    validation_y_datagen = ImageDataGenerator(**validation_y_datagen_no_aug)
    
    validation_x_generator = validation_x_datagen.flow_from_directory(
        path_to_validation_x,
        target_size=(input_size[0], input_size[1]),
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        seed=seed)
    
    validation_y_generator = validation_y_datagen.flow_from_directory(
        path_to_validation_y,
        target_size=(input_size[0], input_size[1]),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)
    
    
    def train_generator(train_x_generator,train_y_generator):
        while True:
            yield train_x_generator.next(),train_y_generator.next()
    
    
    def validation_generator(validation_x_generator,validation_y_generator):
        while True:
            yield validation_x_generator.next(),validation_y_generator.next()
    
    
    print('Data generators ready~')
    
#    loss = 'dice'
    name = 'unet'
    timing = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
    unet = cnn.Unet()
    unet.initiate(input_size[0],input_size[1],input_size[2],loss)
    
    path_to_model = r'./model/'
    path_to_csv = r'./log/'
    path_to_tensorboard = r'./tensorboard/'
    
    checkpoint = ModelCheckpoint(path_to_model+name+'_'+loss+'_'+str(input_size[0])+'_'+'no_aug'+'_'+timing+'.hdf5', monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(path_to_csv+name+'_'+loss+'_'+str(input_size[0])+'_'+'no_aug'+'_'+timing+'.csv')
    unet.model.fit_generator(
            train_generator(train_x_generator,train_y_generator),
            steps_per_epoch=train_step,
            epochs=epochs,
            validation_data=validation_generator(validation_x_generator,validation_y_generator),
            validation_steps=validation_step,
            callbacks=[checkpoint,csv_logger])

