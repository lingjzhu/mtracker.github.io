#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:46:52 2018

@author: luke
"""

import data_generator as dg
import os
import pandas as pd
import skimage.io
import re
import numpy as np


# specify input path
path = r'/media/luke/LENOVO/training_data'
# specify output path
output_path = r'/home/luke/Videos/ultrasound/data/x/'

# load human annotations
labels = pd.read_csv(os.path.join(path,'nn_splines.csv'),sep=',',header='infer')


# select frames that are annoated by subject
participants = labels.participant.drop_duplicates().tolist()
subject = participants[0]
participants = participants[1:]

for subject in participants:
    # select data from a participant
    individual_data = labels.loc[labels['participant']==subject]
    # extract frame indices
    frames = individual_data.drop_duplicates(subset='vframe')['vframe']
    # sample half of the frames
    samples = frames.sample(int(len(frames)/2)).tolist()
    
    # match the associated video
    video_name = subject+"_us.mp4"
    # load video
    video = dg.load_video(os.path.join(path,video_name))
    # extract sample frames
    dg.extract_frames(video,samples,subject,output_path)
    
    print(subject+" done!")


# crop a section of the image 
input_path = r'/home/luke/Videos/ultrasound/data/x/'
output_path = r'/home/luke/Videos/ultrasound/data/x_processed/'

file_list = os.listdir(input_path)

for file in file_list: 
    dg.crop_tongue(input_path,file, output_path)
    
    
# generate masks
input_path = r'/home/luke/Videos/ultrasound/data/x_processed/'

output_path = r'/home/luke/Videos/ultrasound/data/y_processed/'

path = r'/media/luke/LENOVO/training_data'

file_list = os.listdir(input_path)

labels = pd.read_csv(os.path.join(path,'nn_splines.csv'),sep=',',header='infer')

dg.generate_splines(file_list,labels,output_path,size=[480,640],up=8)

mask_list = os.listdir(output_path)
for mask in mask_list: 
    dg.crop_tongue(output_path,mask,output_path)

# sanity check

x = np.load(r'/home/luke/Videos/ultrasound/data/training.npy')
y10 = np.load(r'/home/luke/Videos/ultrasound/data/y_10.npy')
i=15900
skimage.io.imshow(np.squeeze(x[i]))
skimage.io.imshow(np.squeeze(y10[i]))
