import numpy as np
import params

'''
This class stores the information for the lane
Based on current and past detections, it finds the best fitting polynomial
'''
class Line():

	def __init__(self):
		# image for poly search
		self.image = []
		# was the line detected in the last iteration?
		self.detected = False
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = np.array([0,0,0], dtype='float')
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		#radius of curvature of the line in some units
		self.radius_of_curvature  = []
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')
		#x location of the first window center form the previous run
		self.center = []
		# number of iteration approved
		self.num_iter_approved = 0
		# number of consecutive bad measurements
		self.num_cons_bad_measurements = 0
		# number of consecutive bad measurements
		self.num_cons_good_measurements = 0

	# Use this function to completely reset the line parameters and start from scratch 
	def reset(self):
		self.num_iter_approved = 0
		self.num_cons_good_measurements = 0
		self.num_cons_bad_measurements = 0
		self.best_fit = None
		self.current_fit = None
		self.detected = False

	# https://www.intmath.com/applications-differentiation/8-radius-curvature.php
	def calc_radii_curvature(self, poly, y):
		A = poly[0]
		B = poly[1]
		R = ((1 + (2*A*y + B)**2)**1.5)/(2*abs(A))
		return R

	# evaluates if current measurements meet given criteria.
	def is_suitable_measurement(self):

		ret = False
		if self.current_fit is not None:
			# calc radii of curvature
			self.radius_of_curvature = self.calc_radii_curvature(self.current_fit, np.max(self.image.shape[0]))
			
			# is radii of curvature acceptable
			acceptable_radii = self.radius_of_curvature < params.ACCEPTABLE_RADII

			# if norm diff in poly coeff acceptable
			self.diffs = np.linalg.norm(self.best_fit - self.current_fit)
			acceptable_poly = self.diffs < params.ACCEPTABLE_POLY_DELTA

			if acceptable_radii and acceptable_poly:
				ret = True

			# print(self.current_fit, self.calc_radii_curvature(self.current_fit, np.max(self.image.shape[0])), self.diffs)
		else:
			ret = False
		
		return ret
	
	# if suitable measurement, update the best fit estimate
	def approve_measurement(self):
		self.num_cons_good_measurements +=1
		self.best_fit = (self.best_fit*self.num_iter_approved + self.current_fit)/(self.num_iter_approved + 1)
		self.num_iter_approved += 1
		self.num_cons_bad_measurements = 0 

	# if note suitable measurement, reject the outlier
	def reject_measurement(self):
		self.num_cons_bad_measurements +=1
		self.num_cons_good_measurements = 0
	
	# Check if enough points around best fit and calculate current fit based on it
	def search_around_best_fit(self):
		# HYPERPARAMETER
		# Choose the width of the margin around the previous polynomial to search
		# The quiz grader expects 100 here, but feel free to tune on your own!
		margin = 100

		# Grab activated pixels
		nonzero = self.image.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		### TO-DO: Set the area of search based on activated x-values ###
		### within the +/- margin of our polynomial function ###
		### Hint: consider the window areas for the similarly named variables ###
		### in the previous quiz, but change the windows to our new search area ###
		
		lane_ind = []
		lane_ind = ((nonzerox > (self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + 
						self.best_fit[2] - margin)) & (nonzerox < (self.best_fit[0]*(nonzeroy**2) + 
						self.best_fit[1]*nonzeroy + self.best_fit[2] + margin)))
		
		# Again, extract left and right line pixel positions
		leftx = nonzerox[lane_ind]
		lefty = nonzeroy[lane_ind] 
		
		# Fit new polynomials
		img_shape = self.image.shape

		if leftx.size != 0 and lefty.size != 0:
			fit = np.polyfit(lefty, leftx, 2)
			ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
			fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
		else:
			fit = None

		# set to current fit
		self.current_fit = fit

		return fit
	
	# returns polynomial given the x, y points of the lane.
	# called after sliding window search
	def get_current_pixels_polyfit(self, x, y):
		self.detected = True
		self.best_fit = np.polyfit(y, x, 2)
		self.num_iter_approved = 1
		return self.best_fit
	
	def get_roc(self):

		ploty = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 25.0/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/800 # meters per pixel in x dimension
		
		# Fit new polynomials to x,y in world space
		y_eval = np.max(ploty)
		xf = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
		fit_world = np.polyfit(ploty*ym_per_pix, xf*xm_per_pix, 2)

		# Calculate the new radii of curvature
		radii = ((1 + (2*fit_world[0]*y_eval*ym_per_pix + fit_world[1])**2)**1.5) / np.absolute(2*fit_world[0])

		return radii








