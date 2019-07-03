#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:50:03 2018

@author: luke
"""

import keras
from keras import backend, layers, models, utils
import cnn_metrics as m
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard
from keras import backend as K

BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')

DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x



def DenseNet(blocks,
             weights='imagenet',
             input_shape=None):
    


    img_input = layers.Input(shape=input_shape)
   

    bn_axis = 3 

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    conv1 = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(conv1)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    dense1 = dense_block(x, blocks[0], name='conv2')
    x = transition_block(dense1, 0.5, name='pool2')
    dense2 = dense_block(x, blocks[1], name='conv3')
    x = transition_block(dense2, 0.5, name='pool3')
    dense3 = dense_block(x, blocks[2], name='conv4')
    x = transition_block(dense3, 0.5, name='pool4')
    dense4 = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(dense4)
    x = layers.Activation('relu', name='relu')(x)


    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(img_input, [x,conv1,dense1,dense2,dense3], name='densenet121')
    # Load weights.
    if weights == 'imagenet':
        if blocks == [6, 12, 24, 16]:
            weights_path = utils.get_file(
                'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                DENSENET121_WEIGHT_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='30ee3e1110167f948a6b9946edeeb738')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model



def u_conv_block(inputs, filters, kernel_size=(3,3), activation='relu',padding='same',pooling=True):
        
        conv = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
        conv = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
        
        if pooling == True:
            pooling = layers.MaxPooling2D(pool_size=(2,2))(conv)
            return conv,pooling
        else:
            return conv



def skip_concatenate(deconv,conv,filters,kernel_size=(2,2),strides=(2,2),padding='same'):
    transposed = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(deconv)
    merge_layer = layers.concatenate([transposed, conv], axis=3)
    return merge_layer



def deconv_block(deconv,concat,filters):
    merged = skip_concatenate(deconv,concat,filters=filters)
    conv = u_conv_block(merged,filters=filters,pooling=False)
    return conv
    

def dense_deconv_block(deconv,concat,growth_rate,name,filters=32):
    merged = skip_concatenate(deconv,concat,filters=filters)
    conv = conv_block(merged,growth_rate,name)
    return conv


def DenseUnet_v1(blocks=[6,12,24,16],weights=None,input_shape=[128,128,3]):    
    dense = DenseNet(blocks,weights=weights,input_shape=[128,128,3])
       
    
    x,conv1,dense1,dense2,dense3 = dense.output
    
    x = deconv_block(x,dense3,filters=512)
    x = deconv_block(x,dense2,filters=256)
    x = deconv_block(x,dense1,filters=128)
    x = deconv_block(x,conv1,filters=64)
    x = deconv_block(x,dense.input,32)
    x = layers.Conv2D(1, (1, 1))(x)
    model = models.Model(dense.input, x)
    return model


def DenseUnet_v2(blocks=[6,12,24,16],weights="imagenet",input_shape=[128,128,3],loss="dice",lamb=5,trainable=True):    
    dense = DenseNet(blocks,weights=weights,input_shape=input_shape)
    if trainable == False:   
        for layer in dense.layers:
            layer.trainable = False
    
    x,conv1,dense1,dense2,dense3 = dense.output
    
    x = dense_deconv_block(x,dense3,blocks[3],name='up_conv1',filters=512)
    x = dense_deconv_block(x,dense2,blocks[2],name='up_conv2',filters=256)
    x = dense_deconv_block(x,dense1,blocks[1],name='up_conv3',filters=128)
    x = dense_deconv_block(x,conv1,blocks[0],name='up_conv4',filters=64)
    x = dense_deconv_block(x,dense.input,blocks[0],name='up_conv6',filters=32)
    
    
    if loss == "mse":
        x = layers.Conv2D(1, (1, 1),name='up_conv7')(x)
        model = models.Model(dense.input, x)
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        
    else:
        x = layers.Conv2D(1, (1, 1), activation='sigmoid',name='up_conv7')(x)
        if loss == "dice":
            model = models.Model(dense.input, x)
            model.compile(optimizer=Adam(lr=1e-4), loss=m.dice_coef_loss,
                       metrics=['accuracy',m.dice_coef,m.precision,m.recall])
            
        elif loss == "class_xentropy":
                model = models.Model(dense.input, x)
                model.compile(optimizer=Adam(lr=1e-4), loss=m.cross_entropy_balanced, 
                              metrics=['accuracy',m.dice_coef,m.precision,m.recall,m.fbeta_score])
                
        if loss == "compound":
            model = models.Model(dense.input, x)
            model.compile(optimizer=Adam(lr=1e-4), loss=m.Compound_loss(lamb=lamb),
                       metrics=['accuracy',m.dice_coef,m.precision,m.recall])
            
        elif loss == "pixel_xentropy":
            weights = layers.Input((input_shape[0], input_shape[1], 1))
            masks = layers.Input((input_shape[0], input_shape[1], 1))
            loss = layers.Lambda(m.p_weighted_binary_loss, output_shape=(input_shape[0], input_shape[1], 1))([x, weights, masks])
            model = models.Model(inputs=[dense.input,weights,masks], outputs=loss)
            model.compile(optimizer=Adam(lr=1e-4), loss=m.identity_loss)
            
    return model

#du = DenseUnet_v2(weights=None,input_shape=[128,128,3],loss = "fbeta")
#du.summary()

