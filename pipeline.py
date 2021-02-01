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
			## TODO
			# # check center to center distance
			# accetable_line_width = np.abs(self.left_lane.center[1] - self.right_lane.center[1]) < params.LANE_WIDTH
			# # add more conditions later

			# if accetable_line_width:
			# 	ret = True
			ret = True

		return ret

	
	def process_image(self, image, showOutput=False, showDebugFigs = False):
		# undistort
		undistort = cal_undistort(image, getCalibMtx(), getCalibDist())

		# threshold
		thresh = applyHSVAndSobelXFilter(undistort, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100), plotVisual=showDebugFigs)

		# find src and dst points and warp
		params.IMSHAPE = image.shape
		src  = params.SRC_POINTS
		dst  = params.DST_POINTS
		warped, _ = warp(thresh, src, dst, plotVisual=showDebugFigs)

		self.left_lane.image = warped
		self.right_lane.image = warped

		if self.left_lane.detected and self.right_lane.detected \
			and self.left_lane.num_cons_bad_measurements < params.MAX_CONS_BAD_MEAS \
			and self.right_lane.num_cons_bad_measurements < params.MAX_CONS_BAD_MEAS:
			self.left_lane.search_around_best_fit()
			self.right_lane.search_around_best_fit()

			# report solution quality
			if self.is_suitable_measurement() == False:
				self.left_lane.reject_measurement()
				self.right_lane.reject_measurement()
			else:
				self.left_lane.approve_measurement()
				self.right_lane.approve_measurement()
			
			left_fit = self.left_lane.best_fit
			right_fit = self.right_lane.best_fit

			# print("left poly search =>", self.left_lane.num_cons_good_measurements, self.left_lane.num_cons_bad_measurements)
			# print("poly poly search =>", self.right_lane.num_cons_good_measurements, self.right_lane.num_cons_bad_measurements)

		else:
			self.left_lane.reset()
			self.right_lane.reset()

			# detect lane lines
			leftx, lefty, rightx, righty, _ = find_lane_pixels(warped)
			left_fit = self.left_lane.get_current_pixels_polyfit(leftx, lefty)
			right_fit = self.right_lane.get_current_pixels_polyfit(rightx, righty)
			# print("slide window search", left_fit, right_fit)
			
		# You now have the best fit polynomial, just need to plot on the image

		lane_lines = plot_lanes(warped, left_fit, right_fit)
		
		# unwrap
		unwrap, _ = warp(lane_lines, dst, src, plotVisual=False)

		# draw on original image
		result = cv2.addWeighted(undistort, 1, unwrap, 0.3, 0)

		#plot
		if showOutput:
			plt.imshow(result)
			plt.show()

		return result

