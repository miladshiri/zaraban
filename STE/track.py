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



def track_single(im1, im2, WS, SS):
    ######
    ### track_single > applying block matching between two frames
    ###                to estimate speckles' displacement
    ### im1 = firt frame
    ### im2 = second frame
    ### WS = Window Size
    ### SS= Search Size
    #######
    
    if type(im1) != np.ndarray:
           print ('Error: Input image1 should be numpy.ndarray')
           return 0
    if type(im2) != np.ndarray:
           print ('Error: Input image2 should be numpy.ndarray')
           return 0
       
    f_size = im1.shape
    f_rows = f_size[0]
    f_cols = f_size[1]

    PD = WS + SS #Paddin
    
    TH = 250  #Speckle Threshold
    
    vectx = np.zeros(f_size)
    vecty = np.zeros(f_size)
    progress = 0
    for row in range(1+PD, f_rows-PD):
        for col in range(1+PD, f_cols-PD):
            progress += 1
            print('{:.3}'.format(progress/((f_rows-2*PD)*(f_cols-2*PD))*100))
            window = im1[row-WS:row+WS,col-WS:col+WS] 
            if np.mean(window) > 0:
    #            print ("row:{}, col:{}".format(row, col))
                match_score = np.zeros((2*SS+1, 2*SS+1))
                cross_col = np.zeros((2*SS+1, 2*SS+1))
    
                for ii in range(-SS, SS+1):
                    for jj in range(-SS, SS+1):
                        patch = im2[row+ii-WS:row+ii+WS, col+jj-WS:col+jj+WS]
    #                    match_score[ii+SS, jj+SS] = np.sum(np.power(window-patch, 2))            
                        cross_col[ii+SS, jj+SS] = np.sum((window-window.mean())*(patch-patch.mean())) / (window.std()*patch.std())
                        match_score = cross_col
                
                match_score = np.nan_to_num(match_score)      
                if match_score[SS, SS] == 0:
                   vectx[row, col] = 0;
                   vecty[row, col] = 0; 
                else:
                    a, b = np.where(match_score == np.max(match_score))
                    vectx[row, col] = a[0] - (SS)
                    vecty[row, col] = b[0] - (SS)
         
    return vectx, vecty
    
            
        
        
        
        