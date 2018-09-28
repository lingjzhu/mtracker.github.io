#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:31:11 2018

@author: luke
"""

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert
import skimage.io
import numpy as np
from scipy import interpolate
from scipy import signal
import pandas as pd
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from skimage.transform import resize
from skimage.color import rgb2gray



def normalize(image,size=[64,128]):
    image = resize(image, size)
    image = image.astype('float32')
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= std
    return image


def crop(original_image,boundary = [97,362,95,507]):
    # load the image
    cropped_image = original_image[boundary[0]:boundary[1],boundary[2]:boundary[3]]
    image = rgb2gray(cropped_image)
    return image        



def initialize(path_to_model):
    import cnn_model as cnn
    model = cnn.Unet()
    model.initiate()
    model.load(path_to_model)
    return model
        


def get_spline(model,image,output_size=[265,412],preprocessing=True,boundary= [97,362,95,507]):
    
    if preprocessing == True:
    # preprocess the image for prediction
        image = crop(image,boundary) # crop the image
        image = normalize(image) # normalize the image
    else:
        pass
    # predict the tongue
    assert image.shape == (64,128)
    
    frame = np.empty([1,64,128,1])
    frame[0,:,:,0] = image
    rough_prediction = model.predict(frame)
    # resize the image
    prediction = resize(np.squeeze(rough_prediction),output_size)
    # skeletonize the rough output
    skeleton = skeletonize(np.where(prediction>0.5,1,0))
    return skeleton


def interpolate_tongue_spline(spline,smooth=True,points=100):
    # extract the spline
    index = np.argwhere(spline==True)
    # sort the spline to be strictly increasing for interpolation
    sorted_ind = index[index[:,1].argsort()]
    # linear interpolation to avoid breaking parts
    f = interpolate.interp1d(sorted_ind[:,1],sorted_ind[:,0],kind='nearest',fill_value='extrapolate')
    # get 100-point representation
    x = np.linspace(sorted_ind[:,1][0],sorted_ind[:,1][-1],points)
    # initialize the spline
    init = np.array([x,f(x)]).T
    # choose whether the tongue should be smoothed by B-splines
    if smooth == False:
        return init
    else:
        spline = interpolate.UnivariateSpline(init[:,0],init[:,1])
        s = np.array([x,spline(x)]).T
        #plt.plot(init[:,0],d(init[:,0]))
        return s

def get_active_contour(init, original_image,smooth=False,bc='fixed',alpha=1,beta=1,wline=30,wedge=1,gamma=1):
    # prepare the original image
    #tongue = crop_image(original_image)
    # predict the tongue using active contour
    contour = active_contour(gaussian(original_image, 1), init[1:], bc=bc,
                           alpha=alpha, beta=beta, w_line=wline, w_edge=wedge, gamma=gamma)
    if smooth == False:    
        return contour
    else:
        x = np.linspace(contour[:,0][0],contour[:,0][-1],100)
        spline = interpolate.UnivariateSpline(contour[:,0],contour[:,1])
        smoothed_contour = np.array([x,spline(x)]).T
        return smoothed_contour
        


def downsample(unique_frame,ground_truth,samples=100):
    # select a spline
    example = ground_truth.loc[ground_truth['uniqueframe']==unique_frame]
    example = example.sort_values(by=['xcoord'])
    #example.reset_index()
    # resample the data to 
    f = interpolate.interp1d(example['xcoord'],example['ycoord'],fill_value='extrapolate')
    # get 100-point representation
    x = np.linspace(example['xcoord'].iloc[0],example['xcoord'].iloc[-1],samples)
    init = np.array([x, f(x)]).T
    
    df = pd.DataFrame(init,columns=['xcoord','ycoord'])
    
    framename = [unique_frame for i in range(samples)]
    df['uniqueframe'] = framename
    return df



def mean_sum_of_distance(contour_1,contour_2):
    
    pairwise_distance = euclidean_distances(contour_1,contour_2)
    distance_sum = np.sum(pairwise_distance.min(axis=0)) + np.sum(pairwise_distance.min(axis=1))
    mean_distance = distance_sum/(len(contour_1)+len(contour_2))
    return mean_distance


def main():
    pass

if __name__ == "__main__":
   main()    
