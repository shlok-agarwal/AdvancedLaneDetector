from helper import *
from line import Line
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

class Pipeline():
	def __init__(self):
		self.left_lane = Line()
		self.right_lane = Line()

	def process_image(self, image, showOutput=False):
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
		leftx, lefty, rightx, righty, _ = find_lane_pixels(warped)
		self.left_lane.add_measurement(leftx, lefty)
		self.right_lane.add_measurement(rightx, righty)
		lane_lines = search_around_poly(warped, self.left_lane.current_fit, self.right_lane.current_fit, plotVisual=False)
		
		# unwrap
		unwrap, _ = warp(lane_lines, dst, src, plotVisual=False)

		# draw on original image
		result = cv2.addWeighted(undistort, 1, unwrap, 1, 0)

		#plot
		if showOutput:
			plt.imshow(result)
			plt.show()

		return result

