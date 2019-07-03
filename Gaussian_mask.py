#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:08:29 2018

@author: luke
"""

import data_generator as dgen
import os
import pandas as pd
import skimage.io
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# specify input path
path = './'
output_path = './y/'
# load human annotations
all_splines = pd.read_csv(os.path.join(path,'nn_splines.csv'),sep=',',header='infer')

filelist = os.listdir(path+"/x")



dgen.generate_masks_in_batch(filelist,all_splines,output_path,size=[480,640],sigma=8)




# specify input path
path = './'
output_path = './test_y/'
# load human annotations
all_splines = pd.read_csv(os.path.join(path,'nn_splines_nwind.csv'),sep=',',header='infer')

filelist = os.listdir(path+"/test_x")



dgen.generate_masks_in_batch(filelist,all_splines,output_path,size=[480,640],sigma=8)



'''
Select 2000 samples
'''
import os
import numpy as np
import shutil

# select samples
path = '/home/luke/Videos/ultrasound/data/x/'
output_path = '/home/luke/Videos/ultrasound/data/samples/x/'

files = os.listdir(path)


idx = np.random.choice(len(files),2000)

samples = [files[i] for i in idx ]

for s in samples:
    shutil.copyfile(path+s,output_path+s)



# load image
path = '/home/luke/Videos/ultrasound/data/samples/x/'

frame = skimage.io.imread(path+samples[550])
skimage.io.imshow(frame)

pattern = re.compile(r'([p|s]\d+)_(\d+)\.jpg')
subject = re.match(pattern,samples[550]).group(1)
frame_index = re.match(pattern,samples[550]).group(2)

spline = all_splines.loc[all_splines['participant']==subject]
spline = spline.loc[spline['vframe']==int(frame_index)]

mask = dgen.get_gaussian_mask(spline,3)
skimage.io.imshow(mask)


'''
miny = int(np.min(all_splines['y_raw']))-20
maxy = int(np.max(all_splines['y_raw']))+20
maxx = int(np.max(all_splines['x_raw']))-20
minx = int(np.min(all_splines['x_raw']))+20

boundary = [miny,maxy,minx,maxx]
'''


mask = mask[miny:maxy,minx:maxx]
mask = resize(mask,[64,64])
mask = mask/np.max(mask)
#mask = resize(mask,[256,256])

frame = frame[miny:maxy,minx:maxx]
frame = resize(frame,[64,64])
frame[:,:,0] = 255*mask

fig, ax = plt.subplots(figsize=(18, 10))
ax.imshow(frame)
ax.imshow(mask)
ax.plot(spline[:, 0], spline[:, 1], '-b', lw=3)