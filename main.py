from helper import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

showDebugPlots = True

#### CALIBRATION ####

#reading in test image
test_calib_img = mpimg.imread('camera_cal/calibration3.jpg')

# get camera matrix
mtx, dist = get_cal_mtx(test_calib_img, 9, 6)

# test on image
dst = cal_undistort(test_calib_img, mtx, dist)

# if showDebugPlots:
# 	# Visualize undistortion
# 	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# 	ax1.imshow(test_calib_img)
# 	ax1.set_title('Original Image', fontsize=30)
# 	ax2.imshow(dst)
# 	ax2.set_title('Undistorted Image', fontsize=30)
# 	plt.show()

########

####  THRESHOLDING ####

# Sobel Operation for edge detection
image = mpimg.imread('test_images/straight_lines1.jpg')
# combined = applySobelOperator(image, showDebugPlots)

# Sobel X and S channel filter ####
combined = applyHSVAndSobelXFilter(image, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100), plotVisual=showDebugPlots)

####  PERSPECTIVE TRANSFORM ####
imshape = image.shape
# src = np.float32([[imshape[1]*.44, imshape[0]*.61],[imshape[1]*0.61, imshape[0]*.61], [imshape[1]*.94, imshape[0]], [imshape[1]*.11, imshape[0]*1]])
# dst = np.float32([[imshape[1]*.11, imshape[0]*.61],[imshape[1]*0.94, imshape[0]*.61], [imshape[1]*.94, imshape[0]], [imshape[1]*.11, imshape[0]*1]])

# src = np.float32([[imshape[1]*.16, imshape[0]*.90],[imshape[1]*0.46, imshape[0]*.55], [imshape[1]*.49, imshape[0]*.55], [imshape[1]*.88, imshape[0]*.90]])
# dst = np.float32([[imshape[1]*.16, imshape[0]*.90],[imshape[1]*0.16, imshape[0]*.55], [imshape[1]*.88, imshape[0]*.55], [imshape[1]*.88, imshape[0]*.90]])

src  = np.float32([[(300,650),(620, 420), (650, 420), (1020, 650)]])
dst  = np.float32([[(300,650),(300, 420), (1020, 420), (1020, 650)]])

unwarp_image = unwarp(image, src, dst, plotVisual=showDebugPlots)
