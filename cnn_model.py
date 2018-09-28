#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:33:41 2018

@author: luke
"""

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard
from keras import backend as K
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras import metrics
import tensorflow as tf

# convert the image to grayscale and then standardize the image
import numpy as np
import cnn_metrics as m


class Unet():
    
    def __init__(self):
        pass
        
    
    # convolutional blocks - two convolutional layers and one maxpooling layer
    def __conv_block(self, inputs, filters, kernel_size=(3,3), activation='relu',padding='same',pooling=True):
        
        conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
        conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
        
        if pooling == True:
            pooling = MaxPooling2D(pool_size=(2,2))(conv)
            return conv,pooling
        else:
            return conv


    # skip-concatenation of earlier layers 
    def __skip_concatenate(self, deconv,conv,filters,kernel_size=(2,2),strides=(2,2),padding='same'):
        transposed = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(deconv)
        merge_layer = concatenate([transposed, conv], axis=3)
        return merge_layer


    # upsampling layers - one deconvolutional layer and two convolutional layers
    def __deconv_block(self, deconv,concat,filters):
        merged = self.__skip_concatenate(deconv,concat,filters=filters)
        conv = self.__conv_block(merged,filters=filters,pooling=False)
        return conv



    def initiate(self,rows=64,cols=128,channels=1):
        
        inputs = Input((rows, cols, channels))
        
        conv1,pool1 = self.__conv_block(inputs,filters=32,pooling=True)
        
        conv2,pool2 = self.__conv_block(pool1,filters=64,pooling=True)
        
        conv3,pool3 = self.__conv_block(pool2,filters=128,pooling=True)
        
        conv4,pool4 = self.__conv_block(pool3,filters=256,pooling=True)
        
        conv5 = self.__conv_block(pool4,filters=512,pooling=False)
        
        conv6 = self.__deconv_block(deconv=conv5,concat=conv4,filters=256)
        
        conv7 = self.__deconv_block(deconv=conv6,concat=conv3,filters=128)
        
        conv8 = self.__deconv_block(deconv=conv7,concat=conv2,filters=64)
        
        conv9 = self.__deconv_block(deconv=conv8,concat=conv1,filters=32)
        
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
        model = Model(inputs=[inputs], outputs=[conv10])
    
        model.compile(optimizer=Adam(lr=1e-4), loss=m.dice_coef_loss, metrics=['accuracy',m.dice_coef,m.precision,m.recall])
        
        self.model = model
        
        return self.model
    
    
    # plot model
    def plot(self,file_path):
        plot_model(self.model, show_shapes=True,to_file=file_path)
        
        
    # load pre-trained model 
    def load(self, path):
        self.model.load_weights(path)
        
        
    # begin training model   
    def train(self,x_train,y_train,path_to_model,path_to_log,batch_size=20,epochs=50,vsplit=0.2):
        
        checkpoint = ModelCheckpoint(path_to_model, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger(path_to_log)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              validation_split=vsplit,
              callbacks=[checkpoint,csv_logger])
        
     # predict new data   
    def predict(self,image):
        
        prediction = self.model.predict(image)
        
        return prediction


    # evaluate model performace
    def evaluate(self,x_text,y_test,batch_size=20):
        print(model.evaluate(x_test,y_test,batch_size=batch_size, verbose=1))

        



def main():
    pass

if __name__ == "__main__":
   main()    
