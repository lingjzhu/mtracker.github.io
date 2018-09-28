#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:13:54 2018

@author: luke
"""

import pandas as pd
import data_generator as dg
import os
import cnn_model as cnn
import skimage.io
import numpy as np
import cnn_predict as cp
import matplotlib.pyplot as plt

import re
pattern = re.compile(r'(s\d)\w')

input_path = r'/home/luke/Videos/ultrasound/src/test/x/'
path_to_test_data = r'/home/luke/Videos/ultrasound/src/test/'
test_data = pd.read_csv(os.path.join(path_to_test_data,'nn_splines_nwind.csv'),sep=',',header='infer')


d = [re.match(pattern,i).group(1) for i in test_data.participant.tolist()]

test_data['speaker'] = pd.Series(d)
test_data['uniqueframe'] = test_data['speaker']+ '_'+ test_data['vframe'].astype(str)

s1a = test_data.loc[test_data.participant=='s1a']
s1b = test_data.loc[test_data.participant=='s1b']
s1c = test_data.loc[test_data.participant=='s1c']
s2a = test_data.loc[test_data.participant=='s2a']
s2b = test_data.loc[test_data.participant=='s2b']
s2c = test_data.loc[test_data.participant=='s2c']

a = pd.concat([s1a,s2a])
b = pd.concat([s1b,s2b])
c = pd.concat([s1c,s2c])


#pred_cnn = pd.read_csv(r'/home/luke/Videos/ultrasound/src/test/cnn_prediction.csv')
#pred_cnn_ac = pd.read_csv(r'/home/luke/Videos/ultrasound/src/test/cnn_ac_wline45_prediction.csv')

uniqueframe = a.uniqueframe.drop_duplicates().tolist()

for v in uniqueframe:
    acontour = a.loc[a.uniqueframe==v]
    bcontour = b.loc[b.uniqueframe==v]
    ccontour = c.loc[c.uniqueframe==v]
    
    nv = v+'.jpg'
    #cnn_contour = pred_cnn.loc[pred_cnn.uniqueframe==nv]
    #cnnac_contour = pred_cnn_ac.loc[pred_cnn_ac.uniqueframe==nv]
    
    img = skimage.io.imread(r'/home/luke/Videos/ultrasound/src/test/x/'+nv)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(acontour['x_raw'], acontour['y_raw'], '-g', lw=3)
    #ax.plot(cnn_contour['x'], cnn_contour['y']+5, '-r', lw=3)
    ax.plot(bcontour['x_raw'], bcontour['y_raw'], '-b', lw=3)
    ax.plot(ccontour['x_raw'], ccontour['y_raw'], '-y', lw=3)
    #ax.plot(cnnac_contour['x'], cnnac_contour['y']+8, '--', lw=3)
    fig.savefig('/home/luke/Videos/ultrasound/src/test/annotations/'+nv,dpi = 300)
    plt.clf()  
    plt.close()  
    print(nv)
