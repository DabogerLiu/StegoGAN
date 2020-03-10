import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from skimage import io, transform, color
from skimage.filters import threshold_otsu
import random
import numpy as np
import pickle
import keras

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils, plot_model

height, width = 32, 32
w_hei, w_wid = 32, 32

batch_size = 128
dataset_len = 50000 # total images

test_percentage = 0.1
test_len = int(dataset_len * test_percentage)
train_len = dataset_len - test_len

### data
from keras.datasets import cifar10
(C_array, _), (_, _) = cifar10.load_data()
from keras.datasets import mnist
(W_array, _), (_, _) = mnist.load_data()

W_array = W_array[0:min(W_array.shape[0],C_array.shape[0])]
C_array = C_array[0:min(W_array.shape[0],C_array.shape[0])]

def get_w_array(train=1):
    np.random.seed(1313)
    np.random.shuffle(W_array)

    if train == 1:
        return W_array[0:train_len,:,:] # W Train set
    else:
        test_set = W_array[train_len:,:,:] # W test set
        np.random.seed(None)
        np.random.shuffle(test_set)
        return test_set

def get_c_array(train=1):
    np.random.seed(1313)
    np.random.shuffle(C_array)

    if train == 1:
        return C_array[0:train_len,:,:,:] # C train Set
    else:
        test_set = C_array[train_len:,:,:,:] # C test Set
        np.random.seed(None)
        np.random.shuffle(test_set)
        return test_set

def get_batch(train=1, batch_size=128):
    cn = get_c_array(train)
    wn = get_w_array(train)
    i_c, i_w = 0, 0
    while True:
        ### C
        if i_c+batch_size >= cn.shape[0]:
            i_c = 0
            np.random.seed(None)
            np.random.shuffle(cn)
            c = cn[np.random.randint(0,cn.shape[0], size=batch_size), :, :, :]
        else:
            c = cn[i_c:i_c+batch_size,:,:,:]
        i_c += batch_size

        c_batch = []
        for each_c in c:
            img_c = (each_c - each_c.min()) / (each_c.max() - each_c.min())
            img_c = transform.resize(img_c, (height, width, 3), mode='reflect')
            c_batch.append(img_c)
        c_batch = np.array(c_batch)
        c_batch = np.reshape(c_batch, [batch_size, height, width, 3])
        # print('c:',c_batch.shape, c_batch.max(), c_batch.min())

        #------------------------------------------------------------------
        ### wm
        if i_w+batch_size >= wn.shape[0]:
            i_w = 0
            np.random.seed(None)
            np.random.shuffle(wn)
            w = wn[np.random.randint(0,wn.shape[0], size=batch_size), :, :]
        else:
            w = wn[i_w:i_w+batch_size,:,:]
        i_w += batch_size

        w_batch = []
        for each_w in w:
                # img_w = color.rgb2gray(each_w)
            img_w = (each_w - each_w.min()) / (each_w.max() - each_w.min())
            img_w = transform.resize(each_w, (w_hei, w_wid), mode='reflect')
            w_batch.append(img_w)
        w_batch = np.array(w_batch)
        w_batch = np.reshape(w_batch, [batch_size, w_hei, w_wid, 1])

        yield (c_batch, w_batch)


### layer / model
from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add, MaxPooling2D, Flatten
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
from keras import optimizers
# from keras.utils import multi_gpu_model

def conv_block1(x, scale, prefix):

    d = K.int_shape(x)
    d = d[-1]

    filters = 32

    ### path #1
    p1 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path1_1x1_conv')(x)

    ### path #2
    p2 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_1x1_conv')(x)
    p2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_3x3_conv')(p2)

    ### path #3
    p3 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_1x1_conv')(x)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv1')(p3)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv2')(p3)

    pc = concatenate([p1, p2, p3], axis=-1, name=prefix + 'path_combine')

    ### res
    pr = Conv2D(d, kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path_combine_conv')(pc)
    out = add([x, pr], name=prefix + 'block_output')

    return out

def conv_block(x, scale, prefix):

    d = K.int_shape(x)
    d = d[-1]

    filters = 32

    ### path #1
    p1 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path1_1x1_conv')(x)

    ### path #2
    p2 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_1x1_conv')(x)
    p2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_3x3_conv')(p2)

    ### path #3
    p3 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_1x1_conv')(x)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv1')(p3)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv2')(p3)

    out = concatenate([p1, p2, p3], axis=-1, name=prefix + 'path_combine')

    return out

import tensorflow as tf

def SSIM_LOSS(y_true , y_pred):
    score=tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return 1-score

def G(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3), scale=1):

    C = Input(shape=in_c, name='C')

    W = Input(shape=in_w, name='W')
    W1 = conv_block(W, scale, prefix='w_en1_')

    G = concatenate([C,W1], axis=-1)
    x = conv_block(G, scale=int(scale*2), prefix='em_en_')

    M = Conv2D(3, kernel_size=(3, 3), padding='same', \
               strides=1, activation='sigmoid', name='M')(x)

    G_model = Model(inputs=[C,W], outputs=M)
    G_model.compile(optimizer='adam', loss= SSIM_LOSS)

    print("===========================")
    print("Model  G:{C,W}->M")
    G_model.summary()

    return G_model

def R(in_m = (height, width, 3), scale = 1):

    M = Input(shape = in_m, name='M')

    x = conv_block(M, scale=int(scale*2), prefix='ex_de_')
    x = conv_block(x, scale, prefix='w_de_')

    W_prime = Conv2D(1, kernel_size=(3, 3), padding='same', \
            strides=1, activation='sigmoid', name='W_prime')(x)

    R_model = Model(inputs=M, outputs=W_prime)
    R_model.compile(optimizer='adam', loss = 'binary_crossentropy')
    
    print("===========================")
    print("Model  R:M->W_prime")
    R_model.summary()

    return R_model

# import scipy.ndimage
# from scipy.ndimage import imread
# from numpy.ma.core import exp
# from scipy.constants.constants import pi
# from model.Discriminator import D1, D2

def D1(input_shape=(height, width, 3)):
    x = Input(shape = input_shape, name='D1_shapes')

    x1 = Conv2D(16, (3,3), name='D1_conv1', activation='relu', padding='same')(x)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x2 = Conv2D(32, (3,3), name='D1_conv2', activation='relu', padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    x3 = Conv2D(64, (3,3), name='D1_conv3', activation='relu', padding='same')(x2)
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)

    x4 = Flatten()(x3)

    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)

    output = Dense(units=2, activation='softmax')(x6)

    model = Model(inputs=x, outputs=output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def D2(M_shape=(height, width, 3), C_shape=(height, width, 3), W_shape=(w_hei, w_wid, 1)):
    I1 = Input(shape = M_shape, name='D2_M_shape')
    I2 = Input(shape = C_shape, name='D2_C_shape')
    I3 = Input(shape = W_shape, name='D2_W_shape')

    x = concatenate([I1, I2], axis=-1)
    x = concatenate([x, I3], axis=-1)

    x1 = Conv2D(16, (3,3), name='D2_conv1', activation='relu', padding='same')(x)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x2 = Conv2D(32, (3,3), name='D2_conv2', activation='relu', padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    x3 = Conv2D(64, (3,3), name='D2_conv3', activation='relu', padding='same')(x2)
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)

    x4 = Flatten()(x3)

    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)

    output = Dense(units=2, activation='softmax')(x6)

    model = Model(inputs=[I1, I2, I3], outputs=output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)

def stegoGAN(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3)):
    G_model = G()
    R_model = R()
    D1_model = D1()
    D2_model = D2()
    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')
    M = G_model([C,W])

    ## models for traning
    #  a. G connected to R
    W_prime = R_model(M)
    GR_model = Model(inputs=[C,W],outputs=[M,W_prime])
    GR_model.compile(optimizer='adam', \
                     loss=[SSIM_LOSS, 'binary_crossentropy'], \
                     loss_weights=[1., 1.]   
                        )
    print("===========================")
    print("Model  GR:CW->M->W_prime")
    GR_model.summary()
    
    # models for traning
    #  b. G connected to D1
    score1_M = D1(M)
    score1_C = D1(C)
    d1_loss = - K.mean(K.log(score1_C + 1e-6)+K.log(1-score1_M+1e-6))

    GD1_model = Model(inputs=[C,W],outputs=score1_M)
    GD1_model.add_loss(d1_loss)
    GD1_model.compile(optimizer='adam' )
    GD1_model.summary()

    # models for traning
    #  c. G connected to D2
    C_shuffle = Lambda(shuffling)(C)
    W_shuffle = Lambda(shuffling)(W)

    score2_t = D2([M,C,W])
    score2_f = D2([M,C_shuffle,W_shuffle])
    d2_loss =  - K.mean(K.log(score2_t + 1e-6) + K.log(1-score2_f+1e-6)) 
    group_real_score = D2([M,C, W])

    GD2_model = Model(inputs=[M,C,W],outputs=group_real_score)
    GD2_model.add_loss(d2_loss)
    GD2_model.compile(optimizer='adam')
    GD2_model.summary()

    return GR_model, GD1_model, GD2_model

def stegoGAN2(in_w=(w_hei, w_wid, 1), in_c=(height, width, 3)):
    G_model = G()
    R_model = R()
    D1_model = D1()
    D2_model = D2()
    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')
    M = G_model([C, W])

    ## models for traning
    #  a. G connected to R
    W_prime = R_model(M)
    GR_model = Model(inputs=[C, W], outputs=[M, W_prime])
    GR_model.compile(optimizer='adam', \
                     loss=[SSIM_LOSS, 'binary_crossentropy'], \
                     loss_weights=[1., 1.]
                     )
    print("===========================")
    print("Model  GR:CW->M->W_prime")
    GR_model.summary()

    #  b. G connected to D1
    score1_M = D1_model(M)
    score1_C = D1_model(C)
    d1_loss = - K.mean(K.log(score1_C + 1e-6) + K.log(1 - score1_M + 1e-6))

    GD1_model = Model(inputs=[C, W], outputs=[score1_M,score1_C])
    GD1_model.add_loss(d1_loss)
    GD1_model.compile(optimizer='adam')

    print("===========================")
    print("Model  GD1:CW->M->D1")
    GD1_model.summary()

    #  c. G connected to D2
    C_shuffle = Lambda(shuffling)(C)
    W_shuffle = Lambda(shuffling)(W)

    score2_t = D2_model([M, C, W])
    score2_f = D2_model([M, C_shuffle, W_shuffle])
    d2_loss = - K.mean(K.log(score2_t + 1e-6) + K.log(1 - score2_f + 1e-6))

    GD2_model = Model(inputs=[C, W], outputs=[score2_t, score2_f])
    GD2_model.add_loss(d2_loss)
    GD2_model.compile(optimizer='adam')

    print("===========================")
    print("Model  GD2:CW->M, MCW->D2")
    GD2_model.summary()

    return GR_model, GD1_model, GD2_model, G_model, R_model

# def model_train(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3)):
#     G_model = G()
#     R_model = R()
#     D1_model = D1()
#     D2_model = D2()
#     C = Input(shape=in_c, name='C')
#     W = Input(shape=in_w, name='W')
#     M = G_model([C,W])
#     W_prime = R_model(M)
#
#     C_shuffle = Lambda(shuffling)(C)
#     W_shuffle = Lambda(shuffling)(W)
#     score2_t = D2([M,C,W])
#     score2_f = D2([M,C_shuffle,W_shuffle])
#     d2_loss =  - K.mean(K.log(score2_t + 1e-6) + K.log(1-score2_f+1e-6))
#     group_real_score = D2([M,C, W])
#     model_train = Model(inputs = [C,W], outputs = group_real_score)
#     model_train.add_loss(d2_loss)
#     model_train.compile(optimizer='adam')
#     model_train.summary()
#     return model_train

def train(epochs=100):

    # model
    GR_model, GD1_model, GD2_model, G_model, R_model = stegoGAN2()

    # data
    itr = get_batch(batch_size = batch_size, train = 1)

    # train
    histoty = []
    steps = int(dataset_len / batch_size)
    for epoch in range(epochs):
        for step in range(steps):
            C, W = itr.__next__()

            GR_loss = GR_model.train_on_batch([C,W], [C,W])
            GD1_loss = GD1_model.train_on_batch([C,W], [])
            GD2_loss = GD2_model.train_on_batch([C,W], [])

            print('GR_loss: ', GR_loss, 'GD1_loss: ', GD1_loss, 'GD2_loss: ', GD2_loss)

        history.append([GR_loss,GD1_loss,GD2_loss])

    G_model.save('/h5files/0_G.h5')
    R_model.save('/h5files/1_R.h5')

    # pickle the history

if __name__ == "__main__":
    print("===============")
    # data
    #itr = get_batch(train=1)
    #for i in range(2):
    #    (c_batch, w_batch) = itr.__next__()
    #    img = c_batch[0]
    #    plt.imshow(np.reshape(img,[height, width, 3]),cmap='gray'); plt.show()
    #    wm = w_batch[0]
    #    plt.imshow(np.reshape(wm,[w_hei, w_wid]),cmap='gray'); plt.show()
    #    print(img.max(),img.min(),wm.max(),wm.min())
    
    # model
    #_, _, _, _, _ = stegoGAN2()
    
    # train
    train(epochs=5)


