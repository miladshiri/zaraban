# -*- coding: utf-8 -*-
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
from STE import speckle


def track_fixed_points(frame1, frame2, WS, SS, model, show_message=False):
    ######
    ### track_fixed_points > Estimating the movement between two frames  
    ###                      for all fixed points 
    ### im1 = firt frame
    ### im2 = second frame
    ### WS = Window Size
    ### SS= Search Size
    #######
    
    if type(frame1) != np.ndarray:
           print ('Error: Input image1 should be numpy.ndarray')
           return 0
    if type(frame2) != np.ndarray:
           print ('Error: Input image2 should be numpy.ndarray')
           return 0
       
    f_size = frame1.shape
    f_rows = f_size[0]
    f_cols = f_size[1]

    PD = WS + SS #Padding
    
    vectx = np.zeros(f_size)
    vecty = np.zeros(f_size)
    score_std = np.zeros(f_size)
    score_mean = np.zeros(f_size)
    score_max = np.zeros(f_size)

    progress = 0
    for row in range(1+PD, f_rows-PD):
        for col in range(1+PD, f_cols-PD):
            progress +=1
            if show_message:
                print('{:.3}'.format(progress/((f_rows-2*PD)*(f_cols-2*PD))*100))
            window = frame1[row-WS:row+WS,col-WS:col+WS]
#            if not np.mean(window) > 0:
#                continue
            feature = speckle.feature_extraction(window.reshape((1, window.shape[0], window.shape[1])))
            label = model.predict(feature)
            
            if label[0] != 1: # If it is not speckle, go to next point.
                continue
            
            match_score = np.zeros((2*SS+1, 2*SS+1))
            cross_col = np.zeros((2*SS+1, 2*SS+1))

            for ii in range(-SS, SS+1):
                for jj in range(-SS, SS+1):
                    patch = frame2[row+ii-WS:row+ii+WS, col+jj-WS:col+jj+WS]
#                    match_score[ii+SS, jj+SS] = np.sum(np.power(window-patch, 2))            
                    if window.std()*patch.std():
                        cross_col[ii+SS, jj+SS] = abs(np.mean((window-window.mean())*(patch-patch.mean()) / (window.std()*patch.std())))
            match_score = cross_col
#                        print (cross_col[ii+SS, jj+SS])
            match_score = np.nan_to_num(match_score)
            score_std[row, col] = match_score.std()
            score_mean[row, col] = match_score.mean()
            score_max[row, col] = match_score.max()

            if match_score[SS, SS] == 1 or match_score[SS, SS] != match_score[SS, SS]:
                vectx[row, col] = 0;
                vecty[row, col] = 0; 
            else:
                a, b = np.where(match_score == np.max(match_score))
                vectx[row, col] = b[0] - (SS)
                vecty[row, col] = a[0] - (SS)
         
    return vectx, vecty
    

        
def track_specified_points(frame1, frame2, markers, WS, SS, model=None, show_message=False):
    ######
    ### track_point > applying block matching to track a point between 
    ###               two frames.
    ### im1 = firt frame
    ### im2 = second frame
    ### markers = inpute initial points
    ### WS = Window Size
    ### SS= Search Size
    #######
    
    if type(frame1) != np.ndarray:
           print ('Error: Input image1 should be numpy.ndarray')
           return 0
    if type(frame2) != np.ndarray:
           print ('Error: Input image2 should be numpy.ndarray')
           return 0
    X, Y = markers   
    f_size = frame1.shape
    f_rows = f_size[0]
    f_cols = f_size[1]
#    counts = markers.shape[0]
    counts = X.shape[0]
    PD = WS + SS #Padding
    
    TH = 0  #Speckle Threshold
    
    vecty = np.zeros(f_size)
    progress = 0
    x_displacements = np.zeros_like(X)
    y_displacements = np.zeros_like(Y)
    
    for i in range(0, counts):
        progress += 1
        if show_message:
            print('{:.3}'.format(progress/(counts)*100))
        col = X[i]
        row = Y[i]
        if col==0 and row==0:
            continue
        
        window = frame1[row-WS:row+WS,col-WS:col+WS]
        if model:
            feature = speckle.feature_extraction(window.reshape((1, window.shape[0], window.shape[1])))
            label = model.predict(feature)
            
            if label[0] != 1: # If it is not speckle, go to next point.
                continue
        
        match_score = np.zeros((2*SS+1, 2*SS+1))
        cross_col = np.zeros((2*SS+1, 2*SS+1))
        
        for ii in range(-SS, SS+1):
            for jj in range(-SS, SS+1):
                patch = frame2[row+ii-WS:row+ii+WS, col+jj-WS:col+jj+WS]
#                    match_score[ii+SS, jj+SS] = np.sum(np.power(window-patch, 2))            
                if window.std()*patch.std():
                    cross_col[ii+SS, jj+SS] = abs(np.mean((window-window.mean())*(patch-patch.mean()) / (window.std()*patch.std())))
        match_score = cross_col
        match_score = np.nan_to_num(match_score)

        if match_score[SS, SS] == 1 or match_score[SS, SS] != match_score[SS, SS]:
           y_displacements[i] = 0;
           x_displacements[i] = 0;
        else:
            a, b = np.where(match_score == np.max(match_score))
            y_displacements[i] = a[0] - (SS)
            x_displacements[i] = b[0] - (SS)
 
    return (x_displacements + X, y_displacements + Y)
    

def track_points_sequential(frames, markers, WS, SS, model=None, show_message=True):
    (oldX, oldY) = markers    
    rows = frames.shape[0]
    cols = oldX.shape[0]
    all_new_x = np.zeros((rows, cols))
    all_new_y = np.zeros((rows, cols))
    all_new_x[0] = oldX.reshape((1, -1))
    all_new_y[0] = oldY.reshape((1, -1))
    
    for i in range(0, rows-1):
        newX, newY = track_specified_points(frames[i], frames[i+1],
                     markers=(oldX, oldY), WS=WS, SS=SS, model=model, show_message=False)
        (oldX, oldY) = (newX, newY)
        all_new_x[i+1] = newX.reshape((1, -1))
        all_new_y[i+1] = newY.reshape((1, -1))
        if show_message:
            print('{:.3}'.format((i+1)/(frames.shape[0]-1)*100))
    return all_new_x, all_new_y
        
        
def track_fixed_sequence(frames, WS, SS, model, show_message=True):
    vectx = np.zeros(frames.shape)
    vecty = np.zeros(frames.shape)
    for i in range(0, frames.shape[0]-1):
        (vectx[i+1], vecty[i+1]) = track_fixed_points(frame1=frames[i],
        frame2=frames[i+1], WS=WS, SS=SS, model=model,
        show_message=False)
        if show_message:
            print('{:.3}'.format((i+1)/(frames.shape[0]-1)*100))
    return vectx, vecty
        