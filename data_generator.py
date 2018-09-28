#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:38:50 2017

@author: luke
"""

import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import av
import os
import shutil
import re
from skimage.transform import resize
from skimage.color import rgb2gray
from praatio import tgio


def load_video(path_to_video):
# load the raw ultrasound video 
	video = av.open(path_to_video)
	return video



def load_splines(path_to_splines):
# load the anotated spline data
    splines = pd.read_csv(path_to_splines, sep=',',header='infer')
# select relevant frames
    frame_indices = splines.drop_duplicates(subset='vframe')['vframe'].tolist()
    return splines, frame_indices



def save_frames(video, index, subject, output_path):

    for frame in video.decode(video=0):
        if frame.index in index:
            frame.to_image().save(output_path+subject+'_'+str(frame.index)+".jpg")
        
    

def crop_tongue (input_path,filename,output_path,boundary=[97,362,95,507]):
    '''
    This function crops the central part of the ultrasound image
    '''
    # load the image
    original_image = skimage.io.imread(input_path+filename)
    cropped_image = original_image[boundary[0]:boundary[1],boundary[2]:boundary[3]]
    resized_image = resize(cropped_image, [64,128])
    image = rgb2gray(resized_image)
    # save the new image
    skimage.io.imsave(output_path+filename,image)        
       
 
def generate_splines(filelist,all_splines,output_path,size=[480,640],up=8):
# size should be a two-dimensional array specifying the width and height of the output image
# This function generates the ground truth
    pattern = re.compile(r'(p\d\d)_(\d+)\.jpg')
    for file in filelist:
        subject = re.match(pattern,file).group(1)
        frame_index = re.match(pattern,file).group(2)
        
        spline = all_splines.loc[all_splines['participant']==subject]
        spline = spline.loc[spline['vframe']==int(frame_index)]
        
        
        new_image = np.zeros(size)
        #map the hand corrected spline to an empty image of the same size
        #   as the training images
        for i in range(spline.shape[0]):
            x = int(spline.iloc[i].x_raw)
            y = int(spline.iloc[i].y_raw)
            # artificially  making it thicker
            new_image[y-up:y,x-2:x+2] = 1
        
        skimage.io.imsave(output_path+file,new_image)



def textgrid_to_frame(path_to_textgrid,sampling_rate=60,tier=0):
    textgrid = tgio.openTextgrid(path_to_textgrid)
    frame_tier = textgrid.tierDict[textgrid.tierNameList[tier]].entryList
    
    frame_index = []
    
    for entry in frame_tier:
        start_frame = np.round(entry[0]*sampling_rate)
        end_frame = np.round(entry[1]*sampling_rate)
        frame_sequences = np.arange(start_frame,end_frame+1,dtype=np.int32)
        
        frame_index += frame_sequences.tolist()
    return frame_index
  
    
