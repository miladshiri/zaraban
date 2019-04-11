import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier

from STE.speckle import Speckle
from STE.speckle import pick_point, patch_extraction, feature_extraction, random_patches, train_test_feature_select
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


samples_num = 200
kernel_width = 5
#samples_num*im_list.shape[0]
#
##### UnSupervised test
#speckle_model = Speckle()
#patches = random_patches(im_list, kernel_width)
#features = feature_extraction(patches)
#speckle_model.fit(features)
#
##### Use Model
#
#X, Y = pick_point(im1, points_size=10)
#patches = patch_extraction(im1, (X, Y), kernel_width)
#features = feature_extraction(patches)
#labels = speckle_model.predict(features)
#speckle_model.visualize((X, Y), labels, image=im1, kernel_width=kernel_width)
#

#
#############################################

features, labels = train_test_feature_select(im_list, 10, kernel_width)
speckle_model = Speckle(method='supervised')
speckle_model.fit(features, labels)

title = 'Please select {} points to detect...'.format(10)
X, Y = pick_point(im1, points_size=10, title=title)
patches = patch_extraction(im1, (X, Y), 5)
features1 = feature_extraction(patches)
ll = speckle_model.predict(features1)
speckle_model.visualize((X, Y), ll, image=im1, kernel_width=kernel_width)

#
#
