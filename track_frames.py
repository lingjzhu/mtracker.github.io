import sys
sys.path.insert(0, './src')

import argparse
import skimage.io as io
import numpy as np
import cnn_model as cnn
import cnn_predict as cp
import pandas as pd
import time
import imageio
import csv
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import DenseNet
from skimage.morphology import skeletonize
from skimage.transform import resize

# please specify the path to input and output files


parser = argparse.ArgumentParser()

parser.add_argument("-i","--path_to_img",dest='img', help="path to the folder containing the demo video",
                    default="./test_data/test_x/")
parser.add_argument("-t","--model_type",dest="t",help="specify which type of model to use (u for unet and du for dense unet)",default="du")
parser.add_argument("-m","--model",dest="m",help="path to the model",default="./model/dense_compound.hdf5")
parser.add_argument("-o","--output_folder",dest="o",help="path to the output folder",default="./demo/output.csv")
parser.add_argument("-f","--figure",dest="f",help="path to save figures",default="./demo/")
parser.add_argument("-b","--boundary",dest="b",nargs=4, help="the boundary for cropping the central portion of images",default=[106,385,121,487])
parser.add_argument("-n","--n_frames",dest="n",help="plot every Nth frame (default 100)",default=100)



args = parser.parse_args()



# please specify the path to input and output files
path_to_img = args.img

path_to_model = args.m

path_to_csv_output = args.o

path_to_figures = args.f
# initiate the model


if args.t == "du":
    model = DenseNet.DenseUnet_v2(weights=None,input_shape=[128,128,3],loss = "compound")
    model.load_weights(path_to_model)
elif args.t == "u":
    model = cnn.Unet()
    model.initiate(128,128,3)
    model.load(path_to_model)
else:
    print("specify which type of model to use (u for unet and du for dense unet)")

# load images 
images = os.listdir(path_to_img)

# initiate an empty dataframe
cnn_prediction = pd.DataFrame(columns=['x','y','uniqueframe'])

# plot every Nth frame
N = int(args.n)

# set the boundary for cropping
boundary = args.b
output_size = [boundary[1]-boundary[0],boundary[3]-boundary[2]]
# get splines

with open(path_to_csv_output, mode='w') as out:
    out = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    out.writerow(['x','y','uniqueframe'])
    for index, image in tqdm(enumerate(images)):
        image = io.imread(os.path.join(path_to_img,image))
        img = cp.crop(image,boundary)
        img = cp.normalize(img,[128,128])
        img = np.expand_dims(img,0)
        pred = model.predict(img)
        pred = np.squeeze(pred)
        img = np.squeeze(img)
        pred = resize(np.squeeze(pred),output_size)
        
        sample = np.where(pred < 0.5,0,1)
        skeleton = skeletonize(sample)
        s = cp.interpolate_tongue_spline(skeleton,smooth=True)
        
        # convert the coordinate back to fit the original image
        s[:,0]=s[:,0]+boundary[2]
        s[:,1]=s[:,1]+boundary[0]
        
        #ac = cp.get_active_contour(s,img,smooth=False)
        for j in range(len(s)):
            out.writerow([s[j,0],s[j,1],index])
        
        # plot every Nth frame for inspection
        if index%N == 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(image, cmap=plt.cm.gray)
            #ax.plot(three['xcoord'], three['ycoord'], '--r', lw=3)
            ax.plot(s[:,0], s[:,1], 'ro', lw=3)
            fig.savefig(path_to_figures+str(index)+'.jpg')
            plt.clf()  
            plt.close() 
