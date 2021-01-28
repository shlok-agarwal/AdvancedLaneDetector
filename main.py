from helper import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

showDebugPlots = True

# Load calibration
with open('camera_calibration.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

####  THRESHOLDING ####

img = mpimg.imread('test_images/straight_lines1.jpg')
image = cal_undistort(img, mtx, dist)

# Sobel X and S channel filter ####
combined = applyHSVAndSobelXFilter(image, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100), plotVisual=showDebugPlots)

####  PERSPECTIVE TRANSFORM ####
imshape = image.shape

width = 130
center_x = imshape[1]//2
src  = np.float32([[center_x - width, 450], [center_x + width, 450], [imshape[1], 650], [0,650]])

src  = np.float32([[580, 445], [700, 445], [1040, 650], [270,650]])
dst  = np.float32([[0,0], [imshape[1], 0], [imshape[1], imshape[0]], [0, imshape[0]]])

unwarp_image = unwarp(image, src, dst, plotVisual=showDebugPlots)
