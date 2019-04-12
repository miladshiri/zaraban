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
from os import listdir
from os.path import isfile, join

def read_frames(path, size=(200, 200)):
    frames = np.array([])
    for file in listdir(path):
        file_format = file.split('.')[-1]
        
        if not isfile(join(path, file)):
            continue
        if file_format not in ['bmp', 'jpg']:
            continue
    
        frame = Image.open(join(path, file))
        frame = np.array(frame.resize(size))
        frame = frame.reshape(1, frame.shape[0], frame.shape[1])
        frames = np.concatenate((frames, frame), axis=0) if frames.size else frame
    return frames