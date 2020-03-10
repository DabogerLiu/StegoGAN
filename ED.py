import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle

from skimage import io, transform, color
from skimage.filters import threshold_otsu
import random
import numpy as np
import os

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda

from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils, plot_model


height, width = 128, 128
w_hei, w_wid = 64, 64

dataset_len = 10500 # total images

test_percentage = 0.1
test_len = int(dataset_len * test_percentage)
train_len = dataset_len - test_len

### data
def get_w_names(path='/data/xin/workspace_x/wmnn/18/images',train=1):
    file_name = os.listdir(path)
    random.seed(3131)
    random.shuffle(file_name)

    if train == 1:
        return file_name[0:train_len]
    else:
        test_set_names = file_name[train_len:]
        random.seed(None)
        random.shuffle(test_set_names)
        return test_set_names

def get_c_names(path='/data/xin/workspace_x/wmnn/18/images', train=1):
    file_name = os.listdir(path)
    random.seed(1313)
    random.shuffle(file_name)

    if train == 1:
        return file_name[0:train_len]
    else:
        test_set_names = file_name[train_len:]
        random.seed(None)
        random.shuffle(test_set_names)
        return test_set_names

def get_batch(path='/data/xin/workspace_x/wmnn/18/images', train=1, batch_size=32):
    n = get_c_names(path, train)
    wn = get_w_names(path, train)
    
    i_c, i_w = 0, 0
    while True:
        
        ### C
        if i_c+batch_size >= len(n):
            i_c = 0
            random.seed(None)
            random.shuffle(n)
            c = np.random.choice(n, batch_size)
        else:
            c = n[i_c:i_c+batch_size]
        i_c += batch_size
        
        c_batch = []
        for each_c in c:
            img_c = io.imread(os.path.join(path, each_c))
            img_c = transform.resize(img_c, (height, width, 3), mode='reflect')
            img_c = (img_c - img_c.min()) / (img_c.max()-img_c.min())
            c_batch.append(img_c)
        c_batch = np.array(c_batch)
        c_batch = np.reshape(c_batch, [batch_size, height, width, 3])
        # print('cover:',c_batch.shape, c_batch.max(), c_batch.min())
        
        #------------------------------------------------------------------
        
        ### W
        if i_w+batch_size >= len(n):
            i_w = 0
            np.random.seed(None)
            np.random.shuffle(wn)
            w = np.random.choice(wn, batch_size)
        else:
            w = wn[i_w:i_w+batch_size]
        i_w += batch_size
        
        w_batch = []
        for each_w in w:
            img_w = io.imread(os.path.join(path, each_w))
            img_w = color.rgb2gray(img_w)
            img_w = transform.resize(img_w, (w_hei, w_wid, 1), mode='reflect')
            img_w = (img_w - img_w.min()) / (img_w.max()-img_w.min())
            w_batch.append(img_w)
        w_batch = np.array(w_batch)
        w_batch = np.reshape(w_batch, [batch_size, w_hei, w_wid, 1])
        # print('wm:',w_batch.shape, w_batch.max(), w_batch.min())
        
        yield (c_batch, w_batch)
# global variables
height, width = 128, 128
w_hei, w_wid = 64, 64

batch_size = 32

dataset_len = 10500 # total images

test_percentage = 0.1
test_len = int(dataset_len * test_percentage)
train_len = dataset_len - test_len

### layer / model
from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add, GlobalAveragePooling2D, \
UpSampling2D, BatchNormalization, LeakyReLU, Activation, AveragePooling2D, MaxPooling2D
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop

LR = LeakyReLU()
LR.__name__ = 'relu'

def conv_block(x, scale,filters, prefix):

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
    pr = Conv2D(d, kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path_combine_conv')(pc)
    out = add([x, pr], name=prefix + 'block_output')
    return out


def conv_block1(x, scale, filters, prefix):  
      
    filters = 16
    dc1 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix+'dc1')(x)
    
    dc2in = concatenate([x, dc1], axis=-1, name=prefix+'dc_combine1')
    dc2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix+'dc2')(dc2in)
    
    dc3in = concatenate([x, dc1, dc2], axis=-1, name=prefix+'dc_combine2') 
    dc3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix+'dc3')(dc3in)
   
    out = concatenate([x, dc1, dc2, dc3], axis=-1, name=prefix+'dc_combine3')

    return out

from keras.applications.vgg16 import VGG16

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=[128,128,3])
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

    
def G(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3), scale=1):

    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')
    
    ### encode W
    W_up = conv_block(W, prefix='w_up0', filters=32, scale= scale)
    W_up = UpSampling2D(size=(2,2), name='w_up')(W_up)
    CW = concatenate([C,W_up], axis=-1)    
    ### embed
    M = conv_block1(CW, prefix='embed0', filters=16,scale=scale)
    M = conv_block1(M, prefix='embed1', filters=32, scale=scale *2)
    M = conv_block1(M, prefix='embed4', filters=16, scale=scale )
    M = Conv2D(3, kernel_size=(1, 1), strides=1, \
                padding='same', name='M_conv')(M)
    M = BatchNormalization(name='M_bn')(M)
    M = Activation('relu', name='M')(M)
    
    G_model = Model(inputs=[C,W], outputs=M, name='G')

    print("===========================")
    print("Model  G:{C,W}->M")
    G_model.summary()

    return G_model

def R(in_m =(height, width, 3), scale = 1):

    M = Input(shape = in_m, name='M')
    
    ### extract
    ex = conv_block1(M, prefix='extract0', filters=32, scale=scale)
    #for i in range(4):
    #    ex = conv_block(ex, prefix='extract'+str(i+1), filters=32*(i+1), scale=scale)
    
    ### decode W
    W_dowm = conv_block1(ex, prefix='w',filters= 32, scale = scale)
    W_dowm = conv_block1(W_dowm, prefix='w_dowm', filters=32, scale=scale)
    W_dowm = AveragePooling2D(pool_size=(2,2), name='w_dowm')(W_dowm)
    
    W_prime = Conv2D(1, kernel_size=(1, 1), strides=1, \
                padding='same', name='Wprime_conv')(W_dowm)
    W_prime = BatchNormalization(name='Wprime_bn')(W_prime)
    W_prime = Activation('sigmoid', name='w_prime')(W_prime)
    
    R_model = Model(inputs=M, outputs=W_prime, name='R')
    
    print("===========================")
    print("Model  R:M->W_prime")
    R_model.summary()

    return R_model


def D1(input_shape=(height, width, 3)):
    x = Input(shape = input_shape, name='D1_shapes')
    x1 = Conv2D(16, (3,3), name='D1_conv1', activation='relu', padding='same')(x)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(32, (3,3), name='D1_conv2', activation='relu', padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(64, (3,3), name='D1_conv3', activation='relu', padding='same')(x2)
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
    x3 = BatchNormalization()(x3)
    x3 = BatchNormalization()(x3)
    x4 = GlobalAveragePooling2D()(x3)
    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)
   
    output = Dense(units=1,  use_bias=False)(x6)

    model = Model(inputs=x, outputs=output)
    print("===========================")
    print("Model  D1:Image->real?")
    model.summary()

    return model

def SSIM_LOSS(y_true , y_pred):
    score=tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return 1-score

def mse(y_true , y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse 
def loss(y_true , y_pred): 
    return  SSIM_LOSS(y_true , y_pred) + mse(y_true , y_pred)

def G_D1_model(in_w=(w_hei, w_wid, 1), in_c=(height, width, 3)):
    #Train Discriminator model
    G_model = G()
    #R_model = R()
    D1_model = D1()
    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')
    G_model.trainable = False

    M = G_model([C,W])

    x_real = C
    x_fake = M

    x_real_score = D1_model(x_real)
    x_fake_score = D1_model(x_fake)

    d1_train_model = Model([C,W],[x_real_score, x_fake_score])

    k = 2
    p = 6
    d_loss = K.mean(x_real_score - x_fake_score)

    real_grad = K.gradients(x_real_score, [x_real])[0]
    fake_grad = K.gradients(x_fake_score, [x_fake])[0]

    real_grad_norm = K.sum(real_grad**2, axis=[1, 2, 3])**(p / 2)
    fake_grad_norm = K.sum(fake_grad**2, axis=[1, 2, 3])**(p / 2)
    grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2

    w_dist = K.mean(x_fake_score - x_real_score)

    d1_train_model.add_loss(d_loss + grad_loss)
    d1_train_model.compile(optimizer=Adam(2e-4, 0.5))
    d1_train_model.metrics_names.append('w_dist')
    d1_train_model.metrics_tensors.append(w_dist)

    G_model.trainable = True
    D1_model.trainable = False

    x1_fake = G_model([C,W])
    x1_fake_score = D1_model(x1_fake)

    G_train_model = Model([C,W], x1_fake_score)

    g_loss = K.mean(x1_fake_score) 
    G_train_model.add_loss(g_loss)
    G_train_model.compile(optimizer=Adam(2e-4, 0.5))

    d1_train_model.summary()
    G_train_model.summary()

    return d1_train_model, G_train_model, G_model, D1_model

    
def train(epochs = 100):
    d1_train_model, G_train_model,G_model, D1_model = G_D1_model()
    itr = get_batch(batch_size = batch_size, train = 1)
    history = []
    steps = int(train_len / batch_size)
    for epoch in range(epochs):
        print(' ')
        print('Epoch: ', epoch+1, '/', epochs)
        for step in range(steps):
            C, W = itr.__next__()

            for j in range(5):
                D_loss = d1_train_model.train_on_batch([C,W],[])
            for j in range(1):
                G_loss = G_train_model.train_on_batch([C,W],[])
            
            
            print('\r', step,'/',steps-1, 'D_loss: ', str(D_loss).ljust(10), \
                'G_loss: ', str(G_loss).ljust(10),\
                end='')
    G_model.save('/home/CVL1/Shaobo/StegoGAN/GD1.h5')


if __name__ == "__main__":
    print("===============")
    train(epochs=30)
    itr = get_batch(train=0)
    test = next(itr)
    img = test[0]
    msk = test[1]

    G = load_model('/home/CVL1/Shaobo/StegoGAN/GD1.h5')

    M = G.predict([img,msk])

    plt.figure(figsize=(8,4))
    n = 8
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(img[i].reshape(128, 128, 3))
    #plt.imshow(M[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(M[i].reshape(128, 128, 3))
    #plt.imshow(M[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



       

    plt.show()
    plt.savefig('/home/CVL1/Shaobo/StegoGAN/GD1.jpg')

