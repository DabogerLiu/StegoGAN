import os

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Conv2D, concatenate, Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.models import load_model
from keras.utils import np_utils
from keras import optimizers
plt.switch_backend('agg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

height, width, M_height, M_width, C_height, C_width, W_height, W_width = 32, 32, 32, 32, 32, 32, 32, 32
channel = 3
M_channel, C_channel = 3, 3
W_channel = 1

def D1(input_shape=(height, width, channel)):
    x = Input(shape = input_shape, name='D1_shapes')
    
    x1 = Conv2D(16, (3,3), name='D1_conv1', activation='relu', padding='same')(x)
    #x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x2 = Conv2D(32, (3,3), name='D1_conv2', activation='relu', padding='same')(x1)
    #x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    x3 = Conv2D(64, (3,3), name='D1_conv3', activation='relu', padding='same')(x2)
    #x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
    
    #x4 = Flatten()(x3)
    x4 = GlobalAveragePooling2D()(x3)
    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)
    
    output = Dense(units=2, activation='softmax')(x6)
    
    model = Model(inputs=x, outputs=output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

def D2(M_shape=(M_height, M_width, M_channel), C_shape=(C_height, C_width, C_channel), W_shape=(W_height, W_width, W_channel)):
    I1 = Input(shape = M_shape, name='D2_M_shape')
    I2 = Input(shape = C_shape, name='D2_C_shape')
    I3 = Input(shape = W_shape, name='D2_W_shape')
    
    x = concatenate([I1, I2], axis=-1)
    x = concatenate([x, I3], axis=-1)
    
    x1 = Conv2D(16, (3,3), name='D2_conv1', activation='relu', padding='same')(x)
    #@x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x2 = Conv2D(32, (3,3), name='D2_conv2', activation='relu', padding='same')(x1)
    #x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    x3 = Conv2D(64, (3,3), name='D2_conv3', activation='relu', padding='same')(x2)
    #x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
    
    #x4 = Flatten()(x3)
    x4 = GlobalAveragePooling2D()(x3)
    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)
    
    output = Dense(units=2, activation='softmax')(x6)
    
    model = Model(inputs=[I1, I2, I3], outputs=output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model
