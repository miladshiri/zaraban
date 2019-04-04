############################################
###                 Zaraban              ### 
###             A python tools for       ###
###         analyzing echocardiograms    ### 
###              written by              ###
###  ----------- Milad Shiri ----------  ###
################## 2019 ####################

import numpy as np
from PIL import Image

path = "E:/MathLab/Python/Projects/Zaraban_project/source/
im1 = Image.open(path + "im1.bmp")
im2 = Image.open(path + "im2.bmp")


f_size = np.array([100, 50])
f_rows = f_size[0]
f_cols = f_size[1]


WS = 3 #window_size
SS = 4 #search_size

PD = WS + SS #Paddin

for row in range(1+PD, f_rows-PD):
    for col in range(1+PD, f_cols-PD)
        window=im1[row-WS:row+WS,col-WS:col+WS]
        
        
        