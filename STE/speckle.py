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
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans


class Speckle:
    def __init__(self, sample_size=100, kernel_radius=5):
        self.sample_size = sample_size
        self.kernel_radius = kernel_radius

    def fit(self, im_list):
        kernel_width = self.kernel_radius
        samples_num = self.sample_size
        
        patches = np.empty((0, kernel_width*2+1, kernel_width*2+1))
        
        for im in im_list:
        
            X = np.random.randint(1+kernel_width, f_size[1]-kernel_width, samples_num)
            Y = np.random.randint(1+kernel_width, f_size[0]-kernel_width, samples_num)
            
            
            for num in range(1, samples_num):
                patch = im[Y[num]-kernel_width:Y[num]+kernel_width+1, X[num]-kernel_width:X[num]+kernel_width+1]
                patches = np.concatenate((patches, patch.reshape((1, patch.shape[0], patch.shape[1]))), axis=0)
                
        patches = patches.reshape(patches.shape[0], -1)
        
        features = np.zeros((patches.shape[0], 5))
        
        features[:, 0] = patches.mean(axis=1)
        features[:, 1] = patches.std(axis=1)
        features[:, 2] = patches.max(axis=1)
        features[:, 3] = patches.min(axis=1)
        features[:, 4] = patches.sum(axis=1)
        
        plt.scatter(features[:, 0], features[:, 1])
        
        model = KMeans(n_clusters=2)
        model.fit(features)
        
        centers = model.cluster_centers_
        if centers[0, 4] > centers[1, 4]:
            speckle_class = 0
        else:
            speckle_class = 1
        
        self.__model = model
        self.__speckle_class = speckle_class
    
    def predict(self, patches):
        patches = patches.reshape(patches.shape[0], -1)
        features = np.zeros((patches.shape[0], 5))
        
        features[:, 0] = patches.mean(axis=1)
        features[:, 1] = patches.std(axis=1)
        features[:, 2] = patches.max(axis=1)
        features[:, 3] = patches.min(axis=1)
        features[:, 4] = patches.sum(axis=1)
        model = self.__model
        labels = model.predict(features)
        if self.__speckle_class == 0:
            labels = np.ones_like(labels) - labels
        return labels
        
#####Fit model


path = "E:/MathLab/EWI_Project/Database/Behnam/4C/original/"
#path = "E:/MathLab/Python/Projects/Zaraban_project/source/track_object/test1/"

im1 = Image.open(path + "im (2).bmp")
im2 = Image.open(path + "im (5).bmp")


im1 = im1.resize((200, 200))
im2 = im2.resize((200, 200))

im1 = np.array(im1)
im2 = np.array(im2)
f_size = im1.shape

im_list = im1.reshape(1, f_size[0], f_size[1])
im_list = np.concatenate((im_list, im2.reshape(1, f_size[0], f_size[1])), axis=0)


samples_num = 50
kernel_width = 7
#samples_num*im_list.shape[0]

speckle_modlel = Speckle()
speckle_modlel.fit(im_list)

#### Use Model
X = np.random.randint(1+kernel_width, f_size[1]-kernel_width, samples_num)
Y = np.random.randint(1+kernel_width, f_size[0]-kernel_width, samples_num)

patches = np.zeros((samples_num, kernel_width*2+1, kernel_width*2+1))

for num in range(1, samples_num):
    patches[num, :, :] = im1[Y[num]-kernel_width:Y[num]+kernel_width+1, X[num]-kernel_width:X[num]+kernel_width+1]


labels = speckle_modlel.predict(patches)


speckles_x = X[labels==1]
speckles_y = Y[labels==1]

hole_x = X[labels==0]
hole_y = Y[labels==0]


imn = im1.copy()
for i in range(1, speckles_x.shape[0]):
    imn = cv2.rectangle(imn, (speckles_x[i]-kernel_width, speckles_y[i]-kernel_width)
    , (speckles_x[i]+kernel_width, speckles_y[i]+kernel_width), (200, 200, 230), thickness=1)

for i in range(1, hole_x.shape[0]):
    imn = cv2.rectangle(imn, (hole_x[i]-kernel_width, hole_y[i]-kernel_width)
    , (hole_x[i]+kernel_width, hole_y[i]+kernel_width), (100, 100, 50), thickness=1)


plt.figure()
plt.imshow(imn)
    


