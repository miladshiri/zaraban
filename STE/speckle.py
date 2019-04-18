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
from sklearn.neighbors import KNeighborsClassifier


def pick_point(frame, points_size=100, title=None):      
    fig = plt.figure()
    plt.imshow(frame)
    plt.title(title)
    points = plt.ginput(points_size)
    plt.close(fig)
    X = []
    Y = []
    for point in points:
        X.append(int(round(point[0])))
        Y.append(int(round(point[1])))
    
    return np.array(X), np.array(Y)


def patch_extraction(frame, points, kernel_radius):
    X = points[0]
    Y = points[1]
    samples_num = X.shape[0]
    patches = np.zeros((samples_num, kernel_radius*2+1, kernel_radius*2+1))
    
    for num in range(0, samples_num):
        patches[num, :, :] = frame[Y[num]-kernel_radius:Y[num]+kernel_radius+1, X[num]-kernel_radius:X[num]+kernel_radius+1]
    return patches


def feature_extraction(patches):
    patches = patches.reshape(patches.shape[0], -1)
    features = np.zeros((patches.shape[0], 5))
    
    features[:, 0] = patches.mean(axis=1)
    features[:, 1] = patches.std(axis=1)
    features[:, 2] = patches.max(axis=1)
    features[:, 3] = patches.min(axis=1)
#    features[:, 4] = patches.sum(axis=1)
    return features


def random_patches(frames, kernel_radius, samples_num=10):
    patches = np.zeros((samples_num, kernel_radius*2+1, kernel_radius*2+1, frames.shape[0]))
    points = np.zeros((samples_num, 2, frames.shape[0]), dtype=np.uint8)
    f_size = frames.shape[1:]
    for i, im in enumerate(frames):
        np.random.seed()
        X = np.random.randint(1+kernel_radius, f_size[1]-kernel_radius, samples_num, )
        Y = np.random.randint(1+kernel_radius, f_size[0]-kernel_radius, samples_num)
        points[:, 0, i] = X
        points[:, 1, i] = Y
        
        patches[:, :, :, i] = patch_extraction(im, (X, Y), kernel_radius)
    return patches, points


def train_test_feature_select(frames, points_size, kernel_radius):
    features1 = np.array([])
    features2 = np.array([])

    for frame in frames:
        title = 'Please select {} points with speckle pattern...'.format(points_size)
        X, Y = pick_point(frame, points_size=points_size, title=title)
        patches = patch_extraction(frame, (X, Y), kernel_radius)
        features1 = np.concatenate((features1, feature_extraction(patches)), axis=0) if features1.size else feature_extraction(patches)
    labels1 = np.ones((features1.shape[0], 1))

    for frame in frames:
        title = 'Please select {} points with non-speckle pattern...'.format(points_size)
        X, Y = pick_point(frame, points_size=points_size, title=title)
        patches = patch_extraction(frame, (X, Y), kernel_radius)
        features2 = np.concatenate((features2, feature_extraction(patches)), axis=0) if features2.size else feature_extraction(patches)
    labels2 = np.zeros((features2.shape[0], 1))

    labels = np.concatenate((labels1, labels2), axis=0)
    features = np.concatenate((features1, features2), axis=0)
    data = np.concatenate((features, labels), axis=1)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


class Speckle:
    def __init__(self, sample_size=100, kernel_radius=5, method='unsupervised'):
        self.sample_size = sample_size
        self.kernel_radius = kernel_radius
        self.method = method
        
    def fit(self, features, labels=None):
#        kernel_width = self.kernel_radius
#        samples_num = self.sample_size
#        f_size = im_list.shape[1:]
        
        if self.method == 'unsupervised':
            model = KMeans(n_clusters=2)
            model.fit(features)
            
            centers = model.cluster_centers_
            if centers[0, 0] > centers[1, 0]:
                speckle_class = 0
            else:
                speckle_class = 1
                
        elif self.method == 'supervised':
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(features, labels)
            speckle_class = 1
            
        self.__model = model
        self.__speckle_class = speckle_class    
        return 1
    
    def predict(self, features):
        if self.method == 'unsupervised':
            model = self.__model
            labels = model.predict(features)
            if self.__speckle_class == 0:
                labels = np.ones_like(labels) - labels
        elif self.method == 'supervised':
            model = self.__model
            labels = model.predict(features)
        else:
            print('Model\'s method is not correct!')
        return labels
     
    
def visualize(points, labels, image=None, kernel_width=5):
    X = points[:, 0]
    Y = points[:, 1]
    speckles_x = X[labels==1]
    speckles_y = Y[labels==1]     
    hole_x = X[labels==0]
    hole_y = Y[labels==0]
    
    im = image.copy()
    for i in range(0, speckles_x.shape[0]):
        cv2.rectangle(im, (speckles_x[i]-kernel_width, speckles_y[i]-kernel_width)
        , (speckles_x[i]+kernel_width, speckles_y[i]+kernel_width), (200, 200, 230), thickness=1)
    
    for i in range(0, hole_x.shape[0]):
        cv2.rectangle(im, (hole_x[i]-kernel_width, hole_y[i]-kernel_width)
        , (hole_x[i]+kernel_width, hole_y[i]+kernel_width), (100, 100, 50), thickness=1)
    
    
    plt.figure()
    plt.imshow(im)
    return 1


def random_point_patch_feature(frames, kernel_width, samples_num=100):
    patches, points = random_patches(frames, kernel_width, samples_num=100)
    features = np.array([])
    for i in range(patches.shape[3]):
        ff = feature_extraction(patches[:, :, :, i])
        ff = ff.reshape(ff.shape[0], ff.shape[1], 1)
        features = np.concatenate((features, ff), axis=2) if features.size else ff
    
    return points, patches, features



def flat(features):
    new = np.array([])
    if features.ndim == 3:
        for i in range(features.shape[2]):
            new = np.concatenate((new, features[:, :, i])) if new.size else features[:, :, i]
        return new
    return features