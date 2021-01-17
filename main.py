from helper import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#### CALIBRATION ####

#reading in test image
test_img = mpimg.imread('camera_cal/calibration3.jpg')

# get camera matrix
mtx, dist = get_cal_mtx(test_img, 9, 6)

# test on image
dst = cal_undistort(test_img, mtx, dist)

# visualize
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

########