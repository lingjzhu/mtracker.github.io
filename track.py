#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:29:54 2018

@author: luke
"""

import skimage.io as io
import numpy as np
#import data_generator as dg
import cnn_predict as cp
import pandas as pd
import time
import imageio
import matplotlib.pyplot as plt
import os


# please specify the path to input and output files
path_to_video = r'C:\Users\zhuji\Videos\src\demo\demo_video.mp4'

path_to_model = r'C:\Users\zhuji\Videos\src\models\model_y10.hdf5'

path_to_csv_output = r'C:\Users\zhuji\Videos\src\demo\output.csv'

path_to_figures = r'C:\Users\zhuji\Videos\src\demo\\'


# initiate the model
model = cp.initialize(path_to_model)

# load video 
video = imageio.get_reader(path_to_video)

# initiate an empty dataframe
cnn_prediction = pd.DataFrame(columns=['x','y','uniqueframe'])

# plot every Nth frame
N = 1

# set the boundary for cropping
boundary = [97,362,95,507]

# get splines
for index, img in enumerate(video):
    t0 = time.time()
    pred = cp.get_spline(model,img,output_size=[265,412],preprocessing=True,boundary=boundary)
    s = cp.interpolate_tongue_spline(pred,points=100)
    
    # convert the coordinate back to fit the original image
    s[:,0]=s[:,0]+boundary[2]
    s[:,1]=s[:,1]+boundary[0]
    
    #ac = cp.get_active_contour(s,img,smooth=False)
    spline = pd.DataFrame(s,columns=['x','y'])
    spline['uniqueframe'] = pd.Series([index for i in range(100)])
    cnn_prediction = cnn_prediction.append(spline)
    cnn_prediction.to_csv(path_to_csv_output)
    
    # plot every Nth frame for inspection
    if index%N == 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        #ax.plot(three['xcoord'], three['ycoord'], '--r', lw=3)
        ax.plot(s[:,0], s[:,1], 'ro', lw=3)
        fig.savefig(path_to_figures+str(index)+'.jpg')
        plt.clf()  
        plt.close() 
    t1 = time.time()
    print((t1-t0))    
