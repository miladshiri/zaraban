# -*- coding: utf-8 -*-
############################################
###                 Zaraban              ### 
###             A python tools for       ###
###         analyzing echocardiograms    ### 
###              written by              ###
###  ----------- Milad Shiri ----------  ###
################## 2019 ####################

from STE import track
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from STE.speckle import pick_point
path = "E:/MathLab/EWI_Project/Database/Behnam/4C/original/"
#path = "E:/MathLab/Python/Projects/Zaraban_project/source/track_object/test1/"

im1 = Image.open(path + "im (4).bmp")
im2 = Image.open(path + "im (7).bmp")

#im1 = Image.open(path + "1.jpg")
#im2 = Image.open(path + "2.jpg")


im1 = im1.resize((200, 200))
im2 = im2.resize((200, 200))

im1 = np.array(im1)
im2 = np.array(im2)

plt.imshow(im1)
    

WS = 12 #window_size
SS = 14 #search_size


#markers = np.array([[210, 155],])
#X = markers[:,0]
#Y = markers[:,1]
X, Y = pick_point(im1, points_size=5)

newX, newY = track.track_point(im1=im1, im2=im2, markers=(X, Y), WS=WS, SS=SS)

plt.imshow(im1, cmap='gray')
plt.plot([X, newX], [Y, newY])
plt.show()
#
#
#imx = np.floor((vectx + SS)*255/8)
#imy = np.floor((vecty + SS)*255/8)
#
#plt.imshow(imx, cmap='plasma')
#plt.show()
#plt.imshow(imy, cmap='plasma')        
#plt.show()        
#
#
#
#imx_color = cv2.applyColorMap(imx.astype(np.uint8), cv2.COLORMAP_RAINBOW)
#imx_color = cv2.blur(imx_color, (5, 5))
#imy_color = cv2.applyColorMap(imy.astype(np.uint8), cv2.COLORMAP_RAINBOW)
#
#image_gray = imx_color.copy()
#image_gray[:, :, 0] = im2
#image_gray[:, :, 1] = im2
#image_gray[:, :, 2] = im2
#
#
#
#im1_ov = cv2.addWeighted(imx_color, .5, image_gray, .5, 0)
#
#plt.imshow(im1_ov)   
#plt.show()   
#plt.imshow(imy_color)        
#plt.show()
#    

