from helper import *
from line import Line
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import params

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

	def is_suitable_measurement(self):
		ret = False
		if self.left_lane.is_suitable_measurement() and self.right_lane.is_suitable_measurement():
			# check center to center distance
			accetable_line_width = np.abs(self.left_lane.center[1] - self.right_lane.center[1]) < params.LANE_WIDTH
			# add more conditions later

			if accetable_line_width:
				ret = True

		return ret

	
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

		# self.left_lane.add_measurement(leftx, lefty)
		# self.right_lane.add_measurement(rightx, righty)
		
		# if self.is_suitable_measurement():
		# 	self.left_lane.approve_measurement()
		# 	self.right_lane.approve_measurement()
		# else:
		# 	self.left_lane.reject_measurement()
		# 	self.right_lane.reject_measurement()

		self.left_lane.best_fit = self.left_lane.get_current_pixels_polyfit(leftx, lefty)
		self.right_lane.best_fit = self.right_lane.get_current_pixels_polyfit(rightx, righty)

		# You now have the best fit polynomial, just need to plot on the image

		lane_lines = plot_lanes(warped, self.left_lane.best_fit, self.right_lane.best_fit)
		
		# unwrap
		unwrap, _ = warp(lane_lines, dst, src, plotVisual=False)

		# draw on original image
		result = cv2.addWeighted(undistort, 1, unwrap, 1, 0)

		#plot
		if showOutput:
			plt.imshow(result)
			plt.show()

		return result

