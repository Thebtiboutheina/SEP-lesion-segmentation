
#-----------------------patch to the datasets----------------
path_data = './DRIVE_datasets_training_testing/'

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