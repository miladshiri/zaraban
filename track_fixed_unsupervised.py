import tools
from STE import track
from STE import speckle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

path = "E:/MathLab/EWI_Project/Database/Behnam/4C/original/"
ff = cv2.imread(path+'im (1).bmp')
frames = tools.read_frames(path, size=(200, 200), pattern="im ({}).bmp")
frames = frames[:25]

kernel_width = 5
search_radius = 4

##UnSupervised
points, patches, features = speckle.random_point_patch_feature(frames, kernel_width, samples_num=100)


speckle_model = speckle.Speckle()
speckle_model.fit(speckle.flat(features))

#points, patches, features = speckle.random_point_patch_feature(frames, kernel_width, samples_num=100)
#i = 6
#labels = speckle_model.predict(features[:, :, i])
#speckle.visualize(points=points[:, :, i], labels=labels, image=frames[i])

##Supervised
#features, labels = speckle.train_test_feature_select(im_list, 30, kernel_width)
#speckle_model = speckle.Speckle(method='supervised')
#speckle_model.fit(features, labels)

#

#(vectx, vecty) = track.track_fixed_points(frame1=frames[0], frame2=frames[4], WS=kernel_width, SS=search_radius, model=speckle_model, show_message=True)
#vect = np.floor((vectx + search_radius) / (search_radius*2) * 255)
#plt.figure()
#plt.imshow(vect)
#
#mapx = cv2.applyColorMap(vect.astype(np.uint8), cv2.COLORMAP_RAINBOW)
#plt.figure()
#plt.imshow(mapx)

(vectx, vecty) = track.track_fixed_sequence(frames, WS=kernel_width, SS=search_radius, model=speckle_model,
    show_message=True)
#vect = vectx
#vs = list(vect.shape)
#vs.append(3)
#mapx = np.zeros(tuple(vs))
#for i, v in enumerate(vect):
#    v = np.floor((v + search_radius) * 255 / (search_radius*2) )
#    v_8 = v.astype(np.uint8)
#    mapx[i, :, :, :] = cv2.applyColorMap(v_8, cv2.COLORMAP_JET)
#

#v = vectx[2, :, :]
#v = vecty[2, :, :]
v = vectx

vrgb = tools.convert2map(v, search_radius)
tools.save_as_video('25mf.avi', vrgb)
#RGB = cv2.resize(RGB, (500, 500))
#cv2.imshow('sd', vbgr[2, :, :, :])
#plt.imshow(vrgb[1,:,:,:])
#plt.imshow(mat)
#plt.imshow(vectx[2])

#vv = vect[2]
#vv = vv.reshape((200, 200, 1))
#mapxx = cv2.applyColorMap(vv.astype(np.uint8), cv2.COLORMAP_RAINBOW)
#plt.figure()
#plt.imshow(mapxx)


#tools.save_as_video('out.avi', vectx)
#frames = vectx
#width = frames.shape[2]
#height = frames.shape[1]
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('qw.avi', fourcc, 2, (width, height))
#for frame in frames:
##    frame = cvtColor(frame, COLOR_GRAY2BGR)
#    frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_RAINBOW)
#    
#    print(frame.shape)
#    out.write(frame)
#
#out.release()
#






