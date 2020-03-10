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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils, plot_model


height, width = 128, 128
w_hei, w_wid = 64, 64

batch_size = 64
dataset_len = 10500 # total cover images

test_percentage = 0.1
test_len = int(dataset_len * test_percentage)
train_len = dataset_len - test_len

### data
def get_wm_names(path='/data/xin/workspace_x/wmnn/18/images',train=1):
    #n = [x for x in os.listdir(path)]
    file_name = os.listdir(path)
    random.seed(3131)
    random.shuffle(file_name)

    if train == 1:
        # train_set_names = 
        return file_name[0:train_len]
    else:
        test_set_names = file_name[train_len:]
        random.seed(None)
        random.shuffle(test_set_names)
        return test_set_names

def get_file_names(path='/data/xin/workspace_x/wmnn/18/images', train=1):
    #n = [x for x in os.listdir(path)]
    file_name = os.listdir(path)
    random.seed(1313)
    random.shuffle(file_name)

    if train == 1:
        # train_set_names = 
        return file_name[0:train_len]
    else:
        test_set_names = file_name[train_len:]
        random.seed(None)
        random.shuffle(test_set_names)
        return test_set_names

def get_batch(path='/data/xin/workspace_x/wmnn/18/images', train=1, batch_size=64):
    n = get_file_names(path, train)
    wn = get_wm_names(path, train)
    
    i_c, i_w = 0, 0
    while True:
        
        ### cover
        if i_c+batch_size >= len(n):
            i_c = 0
            random.seed(None)
            random.shuffle(n)
            c = np.random.choice(n, batch_size)
        else:
            c = n[i_c:i_c+batch_size]
        i_c += batch_size
        
        img_batch = []
        for each_c in c:
            img_c = io.imread(os.path.join(path, each_c))
            img_c = transform.resize(img_c, (height, width, 3), mode='reflect')
            img_batch.append(img_c)
        img_batch = np.array(img_batch)
        img_batch = np.reshape(img_batch, [batch_size, height, width, 3])
        # print('cover:',img_batch.shape, img_batch.max(), img_batch.min())
        
        #------------------------------------------------------------------
        
        ### wm
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
            img_w = transform.resize(img_w, ( w_hei, w_wid, 1), mode='reflect')
            w_batch.append(img_w)
        w_batch = np.array(w_batch)
        w_batch = np.reshape(w_batch, [batch_size, w_hei, w_wid, 1])
        # print('wm:',w_batch.shape, w_batch.max(), w_batch.min())
        
        yield (img_batch, w_batch)


### layer / model

from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D, Reshape
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
from keras import optimizers
# from keras.utils import multi_gpu_model

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


def conv_block1(x, scale, prefix):  
      
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


import tensorflow as tf

def SSIM_LOSS(y_true , y_pred):
    score=tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return 1-score

def w_enc(x, scale, prefix):
    
    ### conv and upsample
    wm_2D = conv_block(x, scale, prefix=prefix + 'conv1_')
    wm_2D = Conv2D(24, kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'up1')(wm_2D)
    wm_2D = conv_block(wm_2D, scale, prefix=prefix + 'conv2_')
    wm_2D = Conv2D(12, kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'up2')(wm_2D)
    
    return wm_2D

def w_dec(x, scale, prefix):

    ### conv and downsample
    m_ext = conv_block1(x, scale, prefix=prefix+'conv1_')
    m_ext = Conv2D(12, kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'pool1')(m_ext)
    m_ext = conv_block1(m_ext, scale, prefix=prefix+'conv2_')
    m_ext = Conv2D(1, kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'pool2')(m_ext)

    return m_ext


def G(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3), scale=1):

    C = Input(shape=in_c, name='C') #128 128 3

    W = Input(shape=in_w, name='W') #64 64 1

    w_code = w_enc(W, scale, 'w_enc_')#12 64 64
    w_code = Reshape(target_shape=(128,128,3), name='wm_code')(w_code) #

    G = concatenate([C,w_code], axis=-1)
    x = conv_block1(G, scale=int(scale*2), prefix='em_en_1')
    x = conv_block1(x, scale=int(scale*2), prefix='em_en_2')


    M = Conv2D(3, kernel_size=(3, 3), padding='same', strides=1, activation='sigmoid', name='M')(x) #128 128 3

    G_model = Model(inputs=[C,W], outputs=M)
    G_model.compile(optimizer='adam', loss= SSIM_LOSS)

    print("===========================")
    print("Model  G:{C,W}->M")
    G_model.summary()

    return G_model

def R(in_m = (height, width, 3), scale = 1):

    M = Input(shape = in_m, name='M')

    M1 = Reshape(target_shape=(64,64,12), name='wm_code1_reshapeb')(M)

    W_prime = w_dec(M1, scale, 'm1_dec_')


    R_model = Model(inputs=M, outputs=W_prime)
    R_model.compile(optimizer='adam', loss = 'binary_crossentropy')
    
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
   
    output = Dense(units=1, activation='sigmoid')(x6)

    model = Model(inputs=x, outputs=output)
    print("===========================")
    print("Model  D1:Image->real?")
    model.summary()

    return model

def D2(M_shape=(height, width, 3), C_shape=(height, width, 3), W_shape=(w_hei, w_wid, 1)):
   
    scale = 1
    I1 = Input(shape = M_shape, name='D2_M_shape')
    I2 = Input(shape = C_shape, name='D2_C_shape')
    I3 = Input(shape = W_shape, name='D2_W_shape')

    x = concatenate([I1, I2], axis=-1)
    w_code = w_enc(I3, scale , 'w_enc_')
    w_code = Reshape(target_shape=(128,128,3), name='wm_code')(w_code)

    x = concatenate([x, w_code], axis=-1)

    x1 = Conv2D(16, (3,3), name='D2_conv1', activation='relu', padding='same')(x)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    #x1 = BatchNormalization()(x1)
    x2 = Conv2D(32, (3,3), name='D2_conv2', activation='relu', padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    #x2 = BatchNormalization()(x2)
    x3 = Conv2D(64, (3,3), name='D2_conv3', activation='relu', padding='same')(x2)
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
    #x3 = BatchNormalization()(x3)

    #x4 = Flatten()(x3)
    x3 = BatchNormalization()(x3)
    x4 = GlobalAveragePooling2D()(x3)

    x5 = Dense(units=512, activation='relu')(x4)
    x6 = Dense(units=256, activation='relu')(x5)
    #x6 = BatchNormalization()(x6)

    output = Dense(units=1, activation='sigmoid')(x6)

    model = Model(inputs=[I1, I2, I3], outputs=output)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("===========================")
    print("Model  D2:three images->group?")
    model.summary()
    return model


def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)

from keras.applications.vgg16 import VGG16

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=[128,128,3])
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def stegoGAN(in_w=(w_hei, w_wid, 1), in_c=(height, width, 3)):
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
    #GR_model.compile(optimizer='adam', \
    #                 loss=[SSIM_LOSS, 'binary_crossentropy'], \
    #                 loss_weights=[1., 1.]
    #                 )
    
    ssim_loss = SSIM_LOSS(C, M)
    w_loss = K.mean(K.binary_crossentropy(W,W_prime))    
    gr_loss = ssim_loss + w_loss
    GR_model.add_loss(gr_loss)
    GR_model.compile(optimizer='adam')
    
    print("===========================")
    print("Model  GR:CW->M->W_prime")
    GR_model.summary()

    #  b. G connected to D1
    score1_M = D1_model(M)
    score1_C = D1_model(C)
    #d1_loss = - K.mean(K.log(score1_C + 1e-6) + K.log(1 - score1_M + 1e-6))
    #d1_loss = - K.sum(K.log(score1_C + 1e-6) + K.log(1 - score1_M + 1e-6))
    d1_loss = perceptual_loss(C, M)


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
    #d2_loss = - K.mean(K.log(score2_t + 1e-6) + K.log(1 - score2_f + 1e-6))
    d2_loss = - K.sum(K.log(score2_t + 1e-6) + K.log(1 - score2_f + 1e-6))

    GD2_model = Model(inputs=[C, W], outputs=[score2_t, score2_f])
    GD2_model.add_loss(d2_loss)
    GD2_model.compile(optimizer='adam')

    print("===========================")
    print("Model  GD2:CW->M, MCW->D2")
    GD2_model.summary()

    return GR_model, GD1_model, GD2_model, G_model, R_model

def train(epochs=100):

    # model
    GR_model, GD1_model, GD2_model, G_model, R_model = stegoGAN()

    # data
    itr = get_batch(batch_size = batch_size, train = 1)

    # train
    history = []
    steps = int(dataset_len / batch_size)
    for epoch in range(epochs):
        for step in range(steps):
            C, W = itr.__next__()
            
            GR_loss = GR_model.train_on_batch([C,W], [])
            GD1_loss = GD1_model.train_on_batch([C,W], [])
            GD2_loss = GD2_model.train_on_batch([C,W], [])
            if step%50 == 0:
                print('Step:', step, 'GR_loss:', GR_loss, 'GD1_loss:', GD1_loss, 'GD2_loss:', GD2_loss)
        print('============================================================================')
        print('Epoch:', epoch, 'GR_loss:', GR_loss, 'GD1_loss:', GD1_loss, 'GD2_loss:', GD2_loss)
        print('============================================================================')
        history.append([GR_loss,GD1_loss,GD2_loss])
    G_model.save('/home/CVL1/Shaobo/StegoGAN/0_G.h5')
    R_model.save('/home/CVL1/Shaobo/StegoGAN/1_R.h5')
    #GR_model.save('/home/CVL1/Shaobo/StegoGAN/GR.h5')
    

    #with open('train_history/history_0_whole.pkl', 'wb') as file_pi:
        #pickle.dump(history.history, file_pi)


import smtplib
from email.mime.text import MIMEText

server = "smtp.gmail.com:587"
user_account = "johnbrown20033@gmail.com"
password = "ko963852"
mailto_list = ["90liushaobo@gmail.com"]

def send_mail(to_list, sub, content):
    me = "python smtp alert " + "<pythonsmtpalert@gmail.com>"
    msg = MIMEText(content)
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = ";".join(mailto_list)
    try:
        s = smtplib.SMTP(server)
        s.starttls()
        s.login(user_account, password)
        s.sendmail(me, to_list, msg.as_string())
        s.close()
        return True
    except Exception as e:
        print(str(e))
        return False

    # pickle the history


if __name__ == "__main__":
    print("===============")
    
    
    train(epochs=5)
    if send_mail(mailto_list, "Training1.py on GPUstation finished", "training1.py"):  
        print("====notification sent.====")
    
        
        
        
        
        
        
