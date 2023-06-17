from __future__ import division
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import * 
import numpy as np     
import tensorflow as tf


def BCDU_net_D3_BN_d(input_size = (48,48,1)):
    N = input_size[0]
    inputs = Input(input_size) 

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1 = BatchNormalization()(conv1)
    merge1= concatenate ([inputs, conv1], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(merge1)


    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    merge2=concatenate([pool1, conv2], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)
     
 
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)
    merge3=concatenate([pool2, conv3], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)
  
    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4_1 = BatchNormalization()(conv4_1)
   
  
    # D2
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4_1) 
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Dropout(0.2)(conv4)    
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4_2)
    conv4_2 = BatchNormalization()(conv4_2)    
    # D3
    merge_dense = concatenate([conv4_2,conv4_1], axis = 3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge_dense) 
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_3 = Dropout(0.2)(conv4_3)    
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4_3)
    drop4_3 = Dropout(0.2)(conv4_3)
    #
   
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(drop4_3)
    print(up6.shape)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(conv3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge6)
    conv6  = BatchNormalization()(conv6 )
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)
    print(conv6.shape)
    #
    conct1=concatenate([up6,conv6], axis=3)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(conct1)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)
    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True)(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge7)
    conv7  = BatchNormalization()(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)
    #
    c2=concatenate([up7, conv7], axis=3)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(c2)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    
    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True)(merge8)    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge8)
    conv8  = BatchNormalization()(conv8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)
    c3=concatenate([up8, conv8], axis=3)
    conv8 = Conv2D(2, 1, activation = 'relu', padding = 'same')(c3)
    conv8 = BatchNormalization()(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)
    model = Model(inputs = inputs, outputs = conv9)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy',metrics=['accuracy'])
   # model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model