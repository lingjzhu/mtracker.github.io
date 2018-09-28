#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:53:49 2018

@author: luke
"""

import numpy as np
import cnn_model as cnn


unet = cnn.Unet()
unet.initiate()
unet.plot(r'/home/luke/Videos/ultrasound/src/model.png')

x = np.load(r'/home/luke/Videos/ultrasound/data/training.npy')
y10 = np.load(r'/home/luke/Videos/ultrasound/data/y_10.npy')

path_to_model_y10 = r'/home/luke/Videos/ultrasound/data/model_y10.hdf5'
path_to_csv_y10 = r'/home/luke/Videos/ultrasound/data/log_y10.csv'

unet = cnn.Unet()
unet.initiate()
unet.train(x,y10,path_to_model_y10,path_to_csv_y10) 