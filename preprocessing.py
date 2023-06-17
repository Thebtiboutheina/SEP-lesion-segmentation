#***************threshold groundtruth********************

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import glob
i = 0
images_crop_expert=[]
path =sorted(glob.glob("/content/drive/My Drive/*.gif"))
for el in range(len(path)):
        print(path[el])
        img = cv2.imread(path[el])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        ret2, th2 = cv2.threshold (np.array(img_gray,dtype=np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        plt.imshow(th2,cmap='gray')
        images_crop_expert.append(th2)

#*******************Save groundtruth***********************
d=1;
for i in range(len(images_crop_expert)):
 filename = "/content/drive/My Drive/0%d_manual_rotate_p4.tif"%d
 cv2.imwrite(filename, images_crop_expert[i])
 i+=1
 d+=1 

#*********************Crop and resize**********************

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import ImageDraw
import glob
from PIL import Image 

images_crop_expert=[]

path =sorted(glob.glob("/content/drive/My Drive/test/*.tif"))

for el in range(len(path)):
        print (path[el])
        #image = Image.open(path[el])
        image = np.array(Image.open(path[el]).convert('L'))
        img = np.array(image)
        print(img.max())
        cropped = img[30:210,30:250]
        img_c = np.array(cropped).astype('uint8')
        image_c=Image.fromarray(img_c)
        image_gg=image_c.resize((256, 256))
        print(img_c.dtype)
        print(img.shape)
        images_crop_expert.append(image_gg)
      
#***********************Save data*************************

d=1;
for i in range(len(images_crop_expert)):
 if d<=9 :
   images_crop_expert[i].save('/content/drive/My Drive/test/00%d_manual.gif'%d,'GIF')
   i+=1
   d+=1
 else :
   images_crop_expert[i].save('/content/drive/My Drive/test/0%d_manual.gif'%d,'GIF')
   i+=1
   d+=1

#************************Skull stripping*******************

import glob

def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       
    rgb_img = cv2.merge([r,g,b])     
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

im_p=[]
p2=sorted(glob.glob('/content/drive/My Drive/*.tif'))
for i in range (len(p2)):
    print(p2[i])
    img = cv2.imread(p2[i])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ShowImage('Brain with Skull',gray,'gray')

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
    ShowImage('Blended', blended, 'bgr')
    ret, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
    brain_mask = markers==largest_component
    brain_out = img.copy()
    brain_out[brain_mask==False] = (0,0,0)
    im_p.append(brain_out)

ShowImage('SKULL',im_p[4],'rgb')

***********************************************************************
import numpy as np
from PIL import Image
import cv2

def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs
