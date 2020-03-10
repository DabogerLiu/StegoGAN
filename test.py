import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from skimage.filters import threshold_otsu
import random
import numpy as np
import pickle
import keras

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils

from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
# from keras.utils import multi_gpu_model

import tensorflow as tf

import numpy
import scipy.ndimage
from scipy.ndimage import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
from model.discriminator import D1, D2
from ED import get_batch

### test data
itr = get_batch(train=0)
test = next(itr)
img = test[0]
msk = test[1]

G = load_model('/home/CVL1/Shaobo/StegoGAN/GD1.h5')

R = load_model('/home/CVL1/Shaobo/StegoGAN/R.h5')
M = G.predict([img,msk])
W_prime = R.predict([img,msk])

plt.figure(figsize=(8,4))
n = 8
for i in range(n):
    ax = plt.subplot(4, n, i+1)
    plt.imshow(img[i].reshape(128, 128, 3))
    #plt.imshow(M[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i+1 + n)
    #plt.imshow(M[i])
    plt.imshow(M[i].reshape(128, 128, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i +1 + 2*n)
    #plt.imshow(msk[i],cmap='gray')
    plt.imshow(msk[i].reshape(64, 64),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i +1 + 3*n)
    #plt.imshow(msk[i],cmap='gray')
    plt.imshow(W_prime[i].reshape(64, 64),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.savefig('/home/CVL1/Shaobo/StegoGAN/10.jpg')
