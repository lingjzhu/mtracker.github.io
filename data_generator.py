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
#import av
import os
import shutil
import re
from skimage.transform import resize
from skimage.color import rgb2gray
#from praatio import tgio

# Batch processing
'''
class Data_Generator:
    
    # intitialize the object
    def __init__(self, name):
        # the variable 'name' should be the name of the participant
        self.name = name
        pass
'''        
        
'''
def load_video(path_to_video):
# load the raw ultrasound video 
	video = av.open(path_to_video)
	return video
'''


def load_splines(path_to_splines):
# load the anotated spline data
    splines = pd.read_csv(path_to_splines, sep=',',header='infer')
# select relevant frames
    frame_indices = splines.drop_duplicates(subset='vframe')['vframe'].tolist()
    return splines, frame_indices


'''
def save_frames(video, index, subject, output_path):

    for frame in video.decode(video=0):
        if frame.index in index:
            frame.to_image().save(output_path+subject+'_'+str(frame.index)+".jpg")
 '''       
    

def crop_tongue (input_path,filename,output_path,boundary=[106,385,121,487],size=[128,128]):
    '''
        boundary=[97,362,95,507]
    '''
    # load the image
    original_image = skimage.io.imread(input_path+filename)
    cropped_image = original_image[boundary[0]:boundary[1],boundary[2]:boundary[3]]
    resized_image = resize(cropped_image,size)
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


'''
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

for frame_index in index:
        spline = splines[splines.uniqueframe==self.name+'_'+str(frame_index)]
        new_image = np.zeros(size)
        #map the hand corrected spline to an empty image of the same size
        #   as the training images
        for i in range(spline.shape[0]):
            x = int(spline.iloc[i].xcoord)
            y = int(spline.iloc[i].ycoord)
            # artificially  making it thicker
            new_image[y-up:y,x-2:x+2] = 1
            # save the new image
        skimage.io.imsave(output_path+self.name+'_'+str(frame_index)+".jpg",new_image)
   
 '''   
    
def superimpose(self, input_path, outout_path):
    # This function can superimpose the ground truth upon the original image for visual inspection
    pattern = re.compile(r'p\d+?_(\d+)_\w+.jpg')
    #loop over all splines
    for frame_index in self.frame_indices:
        spline = self.splines[self.splines.vframe==frame_index]
        # load the original image
        file_name = re.match(pattern, self.image_list)
        original_image = skimage.io.imread(input_path+file_name)
    
    # map the tongue spline onto the original image
        for i in range(tongue_coordinates.shape[0]):
        #x = int(tongue_coordinates.iloc[i].X)
        #y = int(tongue_coordinates.iloc[i].Y)
            x = int(tongue_coordinates.iloc[i].xcoord)
            y = int(tongue_coordinates.iloc[i].ycoord)
            original_image[y-8:y,x-2:x+2,0] = 255
            original_image[y-8:y,x-2:x+2,1] = 0
            original_image[y-8:y,x-2:x+2,2] = 0
            # save the new image
        skimage.io.imsave(output_path+self.name+"_"+str(frame_index)+'_superim.jpg',original_image)
 

            
def gaussian_k(x0,y0,sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    
    
    
def get_gaussian_mask(annotation,sigma=5, width=640, height=480, scale=True):
    """
        Generate a heatmap with Gaussian distributed contour points
    """
    hm = np.zeros([height,width])
    for i in range(len(annotation)):
        point = gaussian_k(annotation['x_raw'].iloc[i],annotation['y_raw'].iloc[i],sigma,width=640,height=480)
        hm += point
    
    if scale == False:
        return hm
    else:
        return np.int32(hm/np.max(hm))
        
        
        
def generate_masks_in_batch(filelist,all_splines,output_path,size=[480,640],sigma=8):
# size should be a two-dimensional array specifying the width and height of the output image
# This function generates the ground truth
    pattern = re.compile(r'([p|s]\d+)_(\d+)\.jpg')
    for file in filelist:
        subject = re.match(pattern,file).group(1)
        frame_index = re.match(pattern,file).group(2)
        
        spline = all_splines.loc[all_splines['participant']==subject]
        spline = spline.loc[spline['vframe']==int(frame_index)]
        
        
        #map the hand corrected spline to an empty image of the same size
        #as the training images
        hm = get_gaussian_mask(spline,sigma,width=size[1], height=size[0])       
        
        skimage.io.imsave(output_path+file,hm)
