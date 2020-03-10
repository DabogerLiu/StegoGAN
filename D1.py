from skimage import io, transform, color
from skimage.filters import threshold_otsu
import random
import numpy as np
import os

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils, plot_model


height, width = 128, 128
w_hei, w_wid = 128, 128

dataset_len = 10500 # total images
batch_size = 32
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


### layer / model
from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add, GlobalAveragePooling2D, \
UpSampling2D, BatchNormalization, LeakyReLU, Activation, AveragePooling2D, MaxPooling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
from keras import optimizers
import tensorflow as tf

LR = LeakyReLU()
LR.__name__ = 'relu'

def conv_block(x, scale,filters, prefix):

    d = K.int_shape(x)
    d = d[-1]

    filters = 32

    ### path #1
    p1 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation=LR, \
                padding='same', name=prefix + 'path1_1x1_conv')(x)

    ### path #2
    p2 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation=LR, \
                padding='same', name=prefix + 'path2_1x1_conv')(x)
    p2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix + 'path2_3x3_conv')(p2)

    ### path #3
    p3 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation=LR, \
                padding='same', name=prefix + 'path3_1x1_conv')(x)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix + 'path3_3x3_conv1')(p3)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix + 'path3_3x3_conv2')(p3)

    out = concatenate([p1, p2, p3], axis=-1, name=prefix + 'path_combine')
    return out


def conv_block1(x, scale, filters, prefix):  
      
    filters = 16
    dc1 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix+'dc1')(x)
    
    dc2in = concatenate([x, dc1], axis=-1, name=prefix+'dc_combine1')
    dc2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix+'dc2')(dc2in)
    
    dc3in = concatenate([x, dc1, dc2], axis=-1, name=prefix+'dc_combine2') 
    dc3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix+'dc3')(dc3in)
   
    out = concatenate([x, dc1, dc2, dc3], axis=-1, name=prefix+'dc_combine3')

    return out

def w_enc(x, scale, prefix):
    
    ### conv and upsample
    wm_2D = conv_block(x, scale, filters=32,prefix=prefix + 'conv1_')
    wm_2D = Conv2D(24, kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix + 'up1')(wm_2D)
    wm_2D = conv_block(wm_2D, scale, filters=32, prefix=prefix + 'conv2_')
    wm_2D = Conv2D(12, kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name=prefix + 'up2')(wm_2D)
    
    return wm_2D

    
def G(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3), scale=1):

    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')
    #C_encode = Conv2D(32, kernel_size=(3, 3), strides=1, activation=LR, \
               # padding='same', name='Encode_C')(C)
    #W_encode = Conv2D(3, kernel_size=(3, 3), strides=1, activation=LR, \
                #padding='same', name='Encode_W')(W)
    
    CW = concatenate([C,W], axis=-1)    
    
    ### embed
    M = conv_block1(CW, prefix='embed0', filters=32,scale=scale)
    M = Conv2D(3, kernel_size=(1, 1), strides=1, \
                padding='same', name='M_conv')(M)
    M = BatchNormalization(name='M_bn')(M)
    M = Activation('sigmoid', name='M')(M)
    
    G_model = Model(inputs=[C,W], outputs=M, name='G')

    print("===========================")
    print("Model  G:{C,W}->M")
    G_model.summary()

    return G_model

def R(in_m =(height, width, 3), scale = 1):

    M = Input(shape = in_m, name='M')
    
    ### extract
    M_encode = Conv2D(32, kernel_size=(3, 3), strides=1, activation=LR, \
                padding='same', name='Encode_M')(M)
    ### decode W
    W_Decode = conv_block1(M_encode, prefix='w',filters= 32, scale = scale)
    #W_dowm = AveragePooling2D(pool_size=(2,2), name='w_dowm')(W_dowm)
    
    W_prime = Conv2D(1, kernel_size=(1, 1), strides=1, \
                padding='same', name='Wprime_conv')(W_Decode)
    W_prime = BatchNormalization(name='Wprime_bn')(W_prime)
    W_prime = Activation('sigmoid', name='w_prime')(W_prime)
    
    R_model = Model(inputs=M, outputs=W_prime, name='R')
    
    print("===========================")
    print("Model  R:M->W_prime")
    R_model.summary()

    return R_model

def SSIM_LOSS(y_true , y_pred):
    score=tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return 1-score

def mse(y_true , y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse 

def loss(y_true , y_pred): 
    loss =  SSIM_LOSS(y_true , y_pred) + mse(y_true , y_pred)
    return loss

def D1(input_shape=(height, width, 3)):
    x = Input(shape = input_shape, name='D1_shapes')

    x1 = Conv2D(16, (3,3), name='D1_conv1', activation=LR, padding='same')(x)
    #x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    #x1 = BatchNormalization()(x1)
    x2 = Conv2D(32, (3,3), name='D1_conv2', activation=LR, padding='same')(x1)
    #x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x2)
    #x2 = BatchNormalization()(x2)
    x3 = Conv2D(64, (3,3), name='D1_conv3', activation=LR, padding='same')(x2)
    #x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
    #x3 = BatchNormalization()(x3)

    #x4 = Flatten()(x3)
    x3 = BatchNormalization()(x3)
    x4 = GlobalAveragePooling2D()(x3)

    x5 = Dense(units=512, activation=LR)(x4)
    x6 = Dense(units=256, activation=LR)(x5)
   
   

    #output = Dense(units=1, activation='sigmoid')(x6)
    output = Dense(units=1, activation='sigmoid')(x6)

    model = Model(inputs=x, outputs=output)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("===========================")
    print("Model  D1:Image->real?")
    model.summary()

    return model



def stegoGAN(in_w=(w_hei, w_wid, 1), in_c=(height, width, 3)):
    G_model = G()
    R_model = R()
    D1_model = D1()

    C = Input(shape=in_c, name='C')
    W = Input(shape=in_w, name='W')

    M = G_model([C, W])
    W_prime = R_model(M)
    GR_model = Model(inputs=[C, W], outputs=[M, W_prime])

    ssim_loss = SSIM_LOSS(C, M)
    w_loss_1 =  SSIM_LOSS(W,W_prime)
    w_loss_2 =K.mean(K.square(W - W_prime))
    gr_loss = ssim_loss + w_loss_1 + w_loss_2 
    GR_model.add_loss(gr_loss)
    
    GR_model.compile(optimizer='adam')
    
    print("===========================")
    print("Model  GR:CW->M->W_prime")
    
    #G_model.trainable = False

    ## models for traning
    #  a. G connected to R
    

    real_score = D1_model(C)
    fake_score = D1_model(M)
    k = 2
    p = 6
    d1_loss = K.mean(real_score - fake_score)

    real_grad = K.gradients(real_score, [C])[0]
    fake_grad = K.gradients(fake_score, [M])[0]
    
    real_grad_norm = K.sum(real_grad**2, axis=[1, 2, 3])**(p / 2)
    fake_grad_norm = K.sum(fake_grad**2, axis=[1, 2, 3])**(p / 2)
    grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2
    w_dist = K.mean(fake_score - real_score)


    GD1_model = Model(inputs=[C, W], outputs=[fake_score, real_score])
    GD1_model.add_loss(d1_loss + grad_loss)
    GD1_model.compile(optimizer='adam')
    #GD1_model.metrics_names.append('w_dist')
    #GD1_model.metrics_tensors.append(w_dist)

    print("===========================")
    print("Model  GD1:CW->M->D1")
    GR_model.summary()
    GD1_model.summary()

    #G_model.trainable = True
    #D1_model.trainable = False
    #M = G_model([C, W])
    


    return GR_model, GD1_model,  G_model, R_model

def train(epochs=5):

    # model
    GR_model, GD1_model, G_model, R_model = stegoGAN()

    # data
    itr = get_batch(batch_size = batch_size, train = 1)

    # train
    history = []
    steps = int(dataset_len / batch_size)
    for epoch in range(epochs):
        for step in range(steps):

            C, W = itr.__next__()
            GD1_loss = GD1_model.train_on_batch([C,W], [])
            GR_loss = GR_model.train_on_batch([C,W], [])

            if step%50 == 0:
                print('Step:', step, 'GR_loss:', GR_loss, 'GD1_loss:', GD1_loss)
        print('============================================================================')
        print('Epoch:', epoch, 'GR_loss:', GR_loss, 'GD1_loss:', GD1_loss)
        print('============================================================================')
        history.append([GR_loss,GD1_loss])
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
    
    
    train(epochs=10)
    if send_mail(mailto_list, "Training1.py on GPUstation finished", "training1.py"):  
        print("====notification sent.====")
    
