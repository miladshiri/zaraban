############################################
###                 Zaraban              ### 
###             A python tools for       ###
###         analyzing echocardiograms    ### 
###              written by              ###
###  ----------- Milad Shiri ----------  ###
################## 2019 ####################

import numpy as np
from PIL import Image
import cv2


from matplotlib import pyplot as plt
path = "E:/MathLab/EWI_Project/Database/Behnam/4C/original/"
im1 = Image.open(path + "im (4).bmp")
im2 = Image.open(path + "im (5).bmp")

im1 = im1.resize((200, 200))
im2 = im2.resize((200, 200))

im1 = np.array(im1)
im2 = np.array(im2)

plt.imshow(im1)



f_size = im1.shape
f_rows = f_size[0]
f_cols = f_size[1]


WS = 4 #window_size
SS = 4 #search_size

PD = WS + SS #Paddin

TH = 250  #Speckle Threshold

vectx = np.zeros(f_size)
vecty = np.zeros(f_size)

for row in range(1+PD, f_rows-PD):
    for col in range(1+PD, f_cols-PD):
        window = im1[row-WS:row+WS,col-WS:col+WS] 
        if np.mean(window) > 0:
#            print ("row:{}, col:{}".format(row, col))
            match_score = np.zeros((2*SS+1, 2*SS+1))
            cross_col = np.zeros((2*SS+1, 2*SS+1))

            for ii in range(-SS, SS+1):
                for jj in range(-SS, SS+1):
                    patch = im2[row+ii-WS:row+ii+WS, col+jj-WS:col+jj+WS]
#                    match_score[ii+SS, jj+SS] = np.sum(np.power(window-patch, 2))            
                    cross_col[ii+SS, jj+SS] = np.sum((window-window.mean())*(patch-patch.mean()) / (window.std()*patch.std()))
                    match_score = cross_col
            
            match_score = np.nan_to_num(match_score)      
            if match_score[SS, SS] == 0:
               vectx[row, col] = 0;
               vecty[row, col] = 0; 
            else:
                a, b = np.where(match_score == np.max(match_score))
                vectx[row, col] = a[0] - (SS)
                vecty[row, col] = b[0] - (SS)
                print ('ok')
     
imx = np.floor((vectx + SS)*255/8)
imy = np.floor((vecty + SS)*255/8)

plt.imshow(imx, cmap='plasma')
plt.show()
plt.imshow(imy, cmap='plasma')        
plt.show()        



imx_color = cv2.applyColorMap(imx.astype(np.uint8), cv2.COLORMAP_RAINBOW)
imx_color = cv2.blur(imx_color, (5, 5))
imy_color = cv2.applyColorMap(imy.astype(np.uint8), cv2.COLORMAP_RAINBOW)

image_gray = imx_color.copy()
image_gray[:, :, 0] = im2
image_gray[:, :, 1] = im2
image_gray[:, :, 2] = im2



im1_ov = cv2.addWeighted(imx_color, .5, image_gray, .5, 0)

plt.imshow(im1_ov)   
plt.show()   
plt.imshow(imy_color)        
plt.show()
    
        
        
        
        
        