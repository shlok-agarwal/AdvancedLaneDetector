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

if showDebugPlots:
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(test_calib_img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

########

#### Sobel Operation for edge detection ####

image = mpimg.imread('test_images/straight_lines1.jpg')
combined = applySobelOperator(image, showDebugPlots)



