
import os
from tensorflow.keras.callbacks import ModelCheckpoint ,LearningRateScheduler,TensorBoard,ReduceLROnPlateau
from tensorflow.keras.utils import  plot_model
import numpy as np

path_data = '/content/drive/My Drive/arch_orig/'
nb_epoch = 10
batch_size=8

print('extracting patches')
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + 'SEP_dataset_imgs_train.hdf5',
    DRIVE_train_groudTruth    = path_data + 'SEP_dataset_groundTruth_train.hdf5',  #masks
    patch_height = 48,
    patch_width  = 48,
    N_subimgs    = 189000,
    inside_FOV = 'True' #select the patches only inside the FOV  (default == True)
)
patches_imgs_train=patches_imgs_train.reshape(189000,48,48,1)
patches_masks_train=patches_masks_train.reshape(189000,48,48,1)

model= OUR_Model(input_size = (48,48,1))
plot_model(model, to_file='/content/drive/My Drive/arch_orig/test_sgd_001/sgd_model.png')   #check how the model looks like
json_string = model.to_json()
open('/content/drive/My Drive/arch_orig/test_sgd_001/sgd_architecture.json', 'w').write(json_string)

print('Training')


mcp_save = ModelCheckpoint(filepath='/content/drive/My Drive/arch_orig/test_sgd_001/best_weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='auto',verbose=1)
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(patches_imgs_train,patches_masks_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=2,
              validation_split= 0.1, callbacks=[mcp_save])