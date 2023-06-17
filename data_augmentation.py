#Data augmentation 

***********Rotation***************************
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import glob
from PIL import ImageDraw
from PIL import Image 
images_rotate90=[]
images_rotatemoins_90=[]
images_rotate15=[]
images_rotatemoins15=[]
images_rotate45=[]
images_rotatemoins45=[]
path=sorted(glob.glob("/content/drive/My Drive/unet_mask_brain/images/patients/expert/*.gif"))

for i in range(len(path)):
        (print (path[i]))
        images = Image.open(path[i])
        rotate90 = images.rotate(90)
        rotatem90 = images.rotate(-90)
        rotate45 = images.rotate(45)
        rotatem45 = images.rotate(-45)
        rotate15 = images.rotate(15)
        rotatem15 = images.rotate(-15)
        images_rotate90.append(rotate90)
        images_rotatemoins_90.append(rotatem90)
        images_rotate45.append(rotate45)
        images_rotatemoins45.append(rotatem45)
        images_rotate15.append(rotate15)
        images_rotatemoins15.append(rotatem15)

************************Flip**********************

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import Image 
images_horz_flip=[]
images_vert_flip=[]

path="/content/drive/My Drive/unet_mask_brain/images/gif_train/"


for path, subdirs, files in os.walk(path):
 for i in range(len(files)):
        (print (i))
        #image = Image.open(el)
        originale_name = files[i][0:2] + "_training.tif"
        print ("originale name: " + originale_name)
        images = Image.open(path + originale_name)
        #horizontal flip
        hoz_flip = images.transpose(Image.FLIP_LEFT_RIGHT) 
        #vertical flip
        vertical_flip = images.transpose(Image.FLIP_TOP_BOTTOM)
        images_horz_flip.append(hoz_flip)
        images_vert_flip.append(hoz_flip)


*************************** transformation affine a gauche et droite *****************************
#transformation gauche

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw
import sys
from PIL import Image 

orig_translate=[]

path="/content/drive/My Drive/unet_mask_brain/images/gif_train/"

for path, subdirs, files in os.walk(path):
 for i in range(len(files)):
        (print (i))
        #image = Image.open(el)
        originale_name = files[i][0:2] + "_training.tif"
        print ("expert name: " + originale_name)
        images = Image.open(path + originale_name)
        a = 1
        b = 0.2
        c = 0 
        d = 0.2
        e = 1
        f = 0 
        img = images.transform(images.size, Image.AFFINE, (a, b, c, d, e, f))
        orig_translate.append(img)


#---------------transformation droit

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw
import sys
from PIL import Image 

orig_translate=[]

path="/content/drive/My Drive/unet_mask_brain/images/gif_train/2016/"

for path, subdirs, files in os.walk(path):
 for i in range(len(files)):
        (print (i))
        #image = Image.open(el)
        originale_name = files[i][0:2] + "_training.tif"
        print ("expert name: " + originale_name)
        images = Image.open(path + originale_name)
        a = 1
        b = -0.1
        c = 0 #left/right (i.e. 5/-5)
        d = -0.1
        e = 1
        f = 0 #up/down (i.e. 5/-5)
        img = images.transform(images.size, Image.AFFINE, (a, b, c, d, e, f))
        orig_translate.append(img)


#---------------------------Translation-------------------

#Translation gauche

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import Image 
import glob


orig_transla=[]

path=sorted(glob.glob("/content/drive/My Drive/unet_mask_brain/images/patients/expert/*.gif"))

for i in range(len(path)):
        print (path[i])
        #image = Image.open(el)
        #originale_name = files[i][0:2] + "_manual1.gif"
        #print ("originale name: " + originale_name)
        images = Image.open(path[i])
        images = np.array(images)
        WIDTH = images.shape[0]
        HEIGHT = images.shape[1]
        for j in range(WIDTH):
           for i in range(HEIGHT):
               if (i < HEIGHT-20):
                  images[j][i] = images[j][i+20]
               elif (i < HEIGHT-1):
                  images[j][i] = 0
        #groundtruth
        images=Image.fromarray((images * 255).astype('uint8'), mode='L') 
        #original
        #images=Image.fromarray(images)
        orig_transla.append(images)

#----------------------Translation haut------------------

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw
import sys
from PIL import Image 
import glob
orig_translate=[]

path=sorted(glob.glob("/content/drive/My Drive/unet_mask_brain/images/patients/expert/*.gif"))

for i in range(len(path)):
        print (path[i])
        images = Image.open(path[i])
        images = np.array(images)
        WIDTH = images.shape[0]
        HEIGHT = images.shape[1]
        for j in range(WIDTH):
           for i in range(HEIGHT):
              if (j < WIDTH - 10 and j > 10):
                  images[j][i] = images[j+10][i]
              else:
                  images[j][i] = 0
        #groundtruth
        images=Image.fromarray((images * 255).astype('uint8'), mode='L')
        #original
        #images=Image.fromarray(images)
        orig_translate.append(images)


#---------------------------Translation Droit

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import ImageDraw
import glob
from PIL import Image 

orig_translate=[]

path=sorted(glob.glob("/content/drive/My Drive/unet_mask_brain/images/patient1/originale/*.tif"))

for i in range(len(path)):
        print (path[i])
        images = Image.open(path [i])
        images = np.array(images)
        WIDTH = images.shape[0]
        HEIGHT = images.shape[1]
        for i in range(HEIGHT, 1, -1):
           for j in range(WIDTH):
              if (i < HEIGHT-10):
                images[j][i] = images[j][i-10]
              elif (i < HEIGHT-1):
                images[j][i] = 0
        #groundtruth
        #images=Image.fromarray((images * 255).astype('uint8'), mode='L')   
        #Original 
        images=Image.fromarray(images)
        orig_translate.append(images)

#*****************************Save data***********************************
d=1;
for i in range(len(orig_translate)):
  orig_transla[i].save('/content/drive/My Drive/unet_mask_brain/images/patient1/translation_droit/originale/%d_trainingçtranf_droit.tif'%d)
  #orig_transla[i].save('/content/drive/My Drive/unet_mask_brain/images/patient1/translation_droit/originale/0%d__trans_droit.gif'%d,'GIF')
  i+=1
  d+=1


#**************************Elastic******************************

import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

import glob
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))
imo=[]
imm=[]
path=sorted(glob.glob("/content/drive/My Drive/unet_mask_brain/images/patients/tiff_expert/*.tiff"))

for i in range(len(path)):
   print(path[i])
   im = cv2.imread(path[i])
   im = np.array(im)
   im_merge_t = elastic_transform(im, im.shape[1] * 2, im.shape[1] * 0.08, im.shape[1] * 0.08)








