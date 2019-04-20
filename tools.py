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
import cv2

def read_frames(path, size=(200, 200), pattern=None):
    frames = np.array([])
    if not pattern:
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
    else:
        i = 0
        while(True):
            i += 1
            p = join(path, pattern.format(i))
            if isfile(p):
                frame = Image.open(p)
                frame = np.array(frame.resize(size))
                frame = frame.reshape(1, frame.shape[0], frame.shape[1])
                frames = np.concatenate((frames, frame), axis=0) if frames.size else frame
            else:
                break
            
    return frames



def save_as_video(path, frames, overlay=False, source=None):
    if frames.ndim == 4:
        frames = np.flip(frames, axis=3)
    width = frames.shape[2]
    height = frames.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 2, (width, height))
    for i, frame in enumerate(frames):
        s = source[i]
        if frame.ndim < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if s.ndim < 3:
            s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        
        if overlay:
            frame = cv2.addWeighted(frame, .5, s, .5, 0)
        out.write(frame)

    out.release()
    return 1


def convert2map(v, search_radius):    
    v = (v + search_radius) / (search_radius*2)
    H = v * 180 * 250/360
    S = np.ones_like(v) * 200
    V = np.ones_like(v) * 160
    vs = list(v.shape)
    vs.append(3)
    vhsv = np.ones(vs)
    vhsv[:, :, :, 0] = H
    vhsv[:, :, :, 1] = S
    vhsv[:, :, :, 2] = V
    vbgr = np.zeros(vhsv.shape, dtype=np.uint8)
    for i in range(vhsv.shape[0]):
        vbgr[i, :, :, :] = cv2.cvtColor(vhsv[i, :, :, :].astype(np.uint8), cv2.COLOR_HSV2BGR)
    vrgb = np.flip(vbgr, axis=3)
    return vrgb
