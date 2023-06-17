%tensorflow_version 1.x

import numpy as np
import configparser
import tensorflow as tf
from tensorflow.keras import  Model
from tensorflow.keras import layers , Input
from tensorflow.keras.layers import concatenate,Conv2D,UpSampling2D,MaxPooling2D,Reshape,ReLU,Permute, Dropout, Add,BatchNormalization, Dropout,Conv2DTranspose
from tensorflow.keras.backend import backend as K
from tensorflow.keras import  activations
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.callbacks import ModelCheckpoint ,LearningRateScheduler
from tensorflow.keras.utils import  plot_model

def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))

    #
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2),data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2),data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2),data_format='channels_first')(conv3)
     #
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2, 2),data_format='channels_first')(conv4)
    #
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    conv5 = BatchNormalization()(conv5)

    up1 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    #
    up2 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    #
    up3 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    #
    up4 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv9)
    conv10 = Reshape((2,patch_height*patch_width))(conv10)
    conv10 = Permute((2,1))(conv10)
    ############
    conv11 = tf.keras.activations.softmax(conv10)

    model = Model(inputs=inputs, outputs=conv11)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    """
   # Open the file
    with open('/content/drive/My Drive/SEP-unet-master (1)/SEP-unet-master/test/summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    """
    return model