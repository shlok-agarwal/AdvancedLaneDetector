from helper import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Load calibration
with open('camera_calibration.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

def getCalibMtx():
	global mtx
	return mtx

def getCalibDist():
	global dist
	return dist

def pipeline(image, showDebugPlots=False):
	# undistort
	undistort = cal_undistort(image, getCalibMtx(), getCalibDist())

	# threshold
	thresh = applyHSVAndSobelXFilter(undistort, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100), plotVisual=False)

	# find src and dst points and warp
	imshape = image.shape
	src  = np.float32([[580, 445], [700, 445], [1040, 650], [270,650]])
	dst  = np.float32([[0,0], [imshape[1], 0], [imshape[1], imshape[0]], [0, imshape[0]]])
	warped, _ = warp(thresh, src, dst, plotVisual=False)

	# detect lane lines
	left_fit, right_fit = fit_polynomial_from_lane_pixels(warped)
	lane_lines = search_around_poly(warped, left_fit, right_fit, plotVisual=False)
	
	# unwrap
	unwrap, _ = warp(lane_lines, dst, src, plotVisual=False)

	#plot
	if showDebugPlots:
		result = cv2.addWeighted(undistort, 1, unwrap, 1, 0)
		plt.imshow(result)
		plt.show()