import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier

from STE import track
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


frames = np.array([])

start=14
end=20
for i in range(start, end+1):
    frame = Image.open(path + "im ({}).bmp".format(i+1))
    frame = np.array(frame.resize((300, 300)))
    frame = frame.reshape(1, frame.shape[0], frame.shape[1])
    frames = np.concatenate((frames, frame), axis=0) if frames.size else frame


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

X, Y = pick_point(frames[0], points_size=10)
patches = patch_extraction(frames[0], (X, Y), kernel_width)
features = feature_extraction(patches)
labels = speckle_model.predict(features[:, :3])
speckle_model.visualize((X, Y), labels, image=frames[0], kernel_width=kernel_width)

speckle_X = X[labels==1]
speckle_Y = Y[labels==1]


### Track

#markers = np.hstack((speckle_X.reshape(-1, 1), speckle_Y.reshape(-1, 1)))
(oldX, oldY) = (speckle_X, speckle_Y)
all_old_x = np.array([])
all_old_y = np.array([])
all_new_x = np.array([])
all_new_y = np.array([])
for i in range(0, len(frames)-1):
    all_old_x = np.hstack((all_old_x, oldX)) if all_old_x.size else oldX
    all_old_y = np.hstack((all_old_y, oldY)) if all_old_y.size else oldY
    newX, newY = track.track_point(im1=frames[i], im2=frames[i+1], markers=(oldX, oldY), WS=kernel_width, SS=SS)
    (oldX, oldY) = (newX, newY)
    all_new_x = np.hstack((all_new_x, newX)) if all_new_x.size else newX
    all_new_y = np.hstack((all_new_y, newY)) if all_new_y.size else newY
    

print (all_new_x==all_old_x)
print (all_new_y==all_old_y)
#plt.imshow(frames[0], cmap='gray')
plt.plot([all_old_x, all_new_x], [all_old_y, all_new_y])
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



