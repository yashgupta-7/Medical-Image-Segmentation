# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:37:52 2020

@author: Administrator
"""

import warnings
warnings.simplefilter('ignore')

from skimage import io
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte
import os
from PIL import Image
import random

img_list = os.listdir("STARE/stare-images/")
patch_size = 256
######################################################################################################
#################################################TRAIN DATA###########################################
######################################################################################################

count = 0
for img in img_list:
    image = Image.open("STARE/stare-images/"+img)
    label = Image.open("STARE/labels-vk/"+img[0:6]+".vk.ppm")
    
    for i in range(5):
        count+=1
        # Random Patch
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        img1 = img1.save("data/training/"+str(count)+"_orig.jpg")
        lab1 = lab1.save("data/manual/"+str(count)+"_orig.jpg")
        
        # Rotation
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        angle = random.choice([90,180,270])
        img1 = img1.rotate(angle)
        lab1 = lab1.rotate(angle)
        img1 = img1.save("data/training/"+str(count)+"_rot.jpg")
        lab1 = lab1.save("data/manual/"+str(count)+"_rot.jpg")
        
        # Adding Noise with sigma=0.05
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        img1 = img1.save("data/training/"+str(count)+"_noise.jpg")
        lab1 = lab1.save("data/manual/"+str(count)+"_noise.jpg")
        img1 = io.imread("data/training/"+str(count)+"_noise.jpg")
        noisyimg1 = random_noise(img1,var=0.05**2)
        io.imsave("data/training/"+str(count)+"_noise.jpg",img_as_ubyte(noisyimg1))
        
######################################################################################################
#################################################TEST DATA############################################
######################################################################################################

count = 0
for img in img_list:
    image = Image.open("STARE/stare-images/"+img)
    label = Image.open("STARE/labels-vk/"+img[0:6]+".vk.ppm")
    
    for i in range(1):
        count+=1
        # Random Patch
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        img1 = img1.save("data/testing/"+str(count)+"_orig.jpg")
        lab1 = lab1.save("data/testing_manual/"+str(count)+"_orig.jpg")
        
        # Rotation
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        angle = random.choice([90,180,270])
        img1 = img1.rotate(angle)
        lab1 = lab1.rotate(angle)
        img1 = img1.save("data/testing/"+str(count)+"_rot.jpg")
        lab1 = lab1.save("data/testing_manual/"+str(count)+"_rot.jpg")
        
        # Adding Noise with sigma=0.05
        w, h = image.size
        x,y =  np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
        img1 = image.crop((x, y, x+patch_size, y+patch_size))
        lab1 = label.crop((x, y, x+patch_size, y+patch_size))
        img1 = img1.save("data/testing/"+str(count)+"_noise.jpg")
        lab1 = lab1.save("data/testing_manual/"+str(count)+"_noise.jpg")
        img1 = io.imread("data/testing/"+str(count)+"_noise.jpg")
        noisyimg1 = random_noise(img1,var=0.05**2)
        io.imsave("data/testing/"+str(count)+"_noise.jpg",img_as_ubyte(noisyimg1))
