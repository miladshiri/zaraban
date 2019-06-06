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
from .. import tools

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


def train_test_feature_select(frames, kernel_radius, points_size):
    features1 = np.array([])
    features2 = np.array([])
    patches1 = np.zeros((points_size, kernel_radius*2+1, kernel_radius*2+1, frames.shape[0]))
    points1 = np.zeros((points_size, 2, frames.shape[0]), dtype=np.uint8)
    labels1 = np.zeros((points_size, frames.shape[0]))
    
    patches2 = np.zeros((points_size, kernel_radius*2+1, kernel_radius*2+1, frames.shape[0]))
    points2 = np.zeros((points_size, 2, frames.shape[0]), dtype=np.uint8)
    labels2 = np.zeros((points_size, frames.shape[0]))
    for i, frame in enumerate(frames):
        title = 'Please select {} points with speckle pattern...'.format(points_size)
        X, Y = pick_point(frame, points_size=points_size, title=title)
        points1[:, 0, i] = X
        points1[:, 1, i] = Y
        patches1[:, :, :, i] = patch_extraction(frame, (X, Y), kernel_radius)
        ff = feature_extraction(patches1[:, :, :, i])
        ff = ff.reshape(ff.shape[0], ff.shape[1], 1)
        features1 = np.concatenate((features1, ff), axis=2) if features1.size else ff
    
    labels1 = np.ones((points_size, frames.shape[0]))

    for frame in frames:
        title = 'Please select {} points with non-speckle pattern...'.format(points_size)
        X, Y = pick_point(frame, points_size=points_size, title=title)
        points2[:, 0, i] = X
        points2[:, 1, i] = Y
        patches2[:, :, :, i] = patch_extraction(frame, (X, Y), kernel_radius)
        ff = feature_extraction(patches2[:, :, :, i])
        ff = ff.reshape(ff.shape[0], ff.shape[1], 1)
        features2 = np.concatenate((features2, ff), axis=2) if features2.size else ff
    
    labels2 = np.zeros((points_size, frames.shape[0]))

    labels = np.concatenate((labels1, labels2), axis=1)
    features = np.concatenate((features1, features2), axis=2)
    points = np.concatenate((points1, points2), axis=2)
    patches = np.concatenate((patches1, patches2), axis=3)
    
#    data = np.concatenate((features, labels), axis=1)
#    np.random.shuffle(data)
    return points, patches, features, labels


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
            np.random.shuffle(features)
            model = KMeans(n_clusters=2)
            model.fit(features)
            
            centers = model.cluster_centers_
            if centers[0, 0] > centers[1, 0]:
                speckle_class = 0
            else:
                speckle_class = 1
                
        elif self.method == 'supervised':
            data = np.concatenate((features, labels), axis=1)
            np.random.shuffle(data)
            features = data[:, :-1]
            labels = data[:, -1]
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
     
    
def overlay_speckle_box(points, labels, im=None, kernel_width=5):
    X = points[:, 0]
    Y = points[:, 1]
    speckles_x = X[labels==1]
    speckles_y = Y[labels==1]     
    hole_x = X[labels==0]
    hole_y = Y[labels==0]

    im = tools.im3d(im)
    for i in range(0, speckles_x.shape[0]):
        cv2.rectangle(im, (speckles_x[i]-kernel_width, speckles_y[i]-kernel_width)
        , (speckles_x[i]+kernel_width, speckles_y[i]+kernel_width), (200, 200, 230), thickness=1)
    
    for i in range(0, hole_x.shape[0]):
        cv2.rectangle(im, (hole_x[i]-kernel_width, hole_y[i]-kernel_width)
        , (hole_x[i]+kernel_width, hole_y[i]+kernel_width), (100, 100, 50), thickness=1)
    
    
    return im


def random_point_patch_feature(frames, kernel_width, samples_num=100):
    patches, points = random_patches(frames, kernel_width, samples_num=100)
    features = np.array([])
    for i in range(patches.shape[3]):
        ff = feature_extraction(patches[:, :, :, i])
        ff = ff.reshape(ff.shape[0], ff.shape[1], 1)
        features = np.concatenate((features, ff), axis=2) if features.size else ff
    
    return points, patches, features



def flat(mat):
    new = np.array([])
    if mat.ndim == 3:
        for i in range(mat.shape[-1]):
            new = np.concatenate((new, mat[:, :, i])) if new.size else mat[:, :, i]
        return new
    elif mat.ndim == 2:
        for i in range(mat.shape[-1]):
            new = np.concatenate((new, mat[:, i])) if new.size else mat[:, i]
        return new.reshape(-1, 1)
    return mat



