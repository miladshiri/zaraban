import numpy as np
from PIL import Image
import cv2
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier

import tools
from STE.track import track_sequence
from STE.speckle import Speckle
from STE.speckle import pick_point, patch_extraction, feature_extraction, random_patches, train_test_feature_select
#####Fit model


path = "E:/MathLab/EWI_Project/Database/Behnam/4C/original/"
#path = "E:/MathLab/Python/Projects/Zaraban_project/source/track_object/test1/"

im1 = Image.open(path + "im (4).bmp")
im2 = Image.open(path + "im (7).bmp")


im1 = im1.resize((200, 200))
im2 = im2.resize((200, 200))

im1 = np.array(im1)
im2 = np.array(im2)


f_size = im1.shape
im_list = im1.reshape(1, f_size[0], f_size[1])
im_list = np.concatenate((im_list, im2.reshape(1, f_size[0], f_size[1])), axis=0)


frames = tools.read_frames(path)
frames = frames[13:40]
samples_num = 200
kernel_width = 5
SS = 6 #search_size

#X, Y = pick_point(frames[4], points_size=5)
#newX, newY = track.track_point(im1=frames[4], im2=frames[7], markers=(X, Y), WS=kernel_width, SS=SS)
#
#print(newX==X)
#print(newY==Y)
#plt.imshow(frames[4], cmap='gray')
#plt.plot([X, newX], [Y, newY])
#plt.show()

#samples_num*im_list.shape[0]
#
#### UnSupervised test
speckle_model = Speckle()
patches = random_patches(frames, kernel_width, samples_num=100)
features = feature_extraction(patches)
speckle_model.fit(features[:, :3])

#### Use Model

X, Y = pick_point(frames[0], points_size=3)
patches = patch_extraction(frames[0], (X, Y), kernel_width)
features = feature_extraction(patches)
labels = speckle_model.predict(features[:, :3])
speckle_model.visualize((X, Y), labels, image=frames[0], kernel_width=kernel_width)

speckle_X = X[labels==1]
speckle_Y = Y[labels==1]

all_new_x, all_new_y = track_sequence(frames, (speckle_X, speckle_Y), 
                                      kernel_width, SS)

plt.plot(all_new_x, all_new_y)
plt.show()
#
#############################################
#
#features, labels = train_test_feature_select(im_list, 20, kernel_width)
#speckle_model = Speckle(method='supervised')
#speckle_model.fit(features, labels)
#
#title = 'Please select {} points to detect...'.format(10)
#X, Y = pick_point(im1, points_size=20, title=title)
#patches = patch_extraction(im1, (X, Y), 5)
#features1 = feature_extraction(patches)
#ll = speckle_model.predict(features1)
#speckle_model.visualize((X, Y), ll, image=im1, kernel_width=kernel_width)


#(vectx, vecty) = track.track_fixed_points(im1=im1, im2=im2, WS=WS, SS=SS)

