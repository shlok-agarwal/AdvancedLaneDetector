import numpy as np
import params

class Line():

	def __init__(self):
		# image for poly search
		self.image = []
		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		#average x values of the fitted line over the last n iterations
		self.bestx = []
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = np.array([0,0,0], dtype='float')
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		#radius of curvature of the line in some units
		self.radius_of_curvature  = []
		#distance in meters of vehicle center from the line
		self.line_base_pos = None
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')
		#x values for detected line pixels
		self.allx = None
		#y values for detected line pixels
		self.ally = None
		#x location of the first window center form the previous run
		self.center = []
		# conversion of pixels to length
		self.pixels_to_length = []
		# number of iteration approved
		self.num_iter_approved = 0
		# number of iteration rejected
		self.num_iter_rejected = 0

	# Use this function to completely reset the line parameters and start from scratch 
	def reset(self):
		self.num_iter_approved = 0
		self.num_iter_rejected = 0

	
	def calc_radii_curvature(self, poly, y):
		A = poly[0]
		B = poly[1]
		R = ((1 + (2*A*y + B)**2)**1.5)/(2*abs(A))
		return R

	def is_suitable_measurement(self):

		ret = False
		if self.current_fit is not None:
			# calc radii of curvature
			self.radius_of_curvature = self.calc_radii_curvature(self.current_fit, np.max(self.ally))
			
			# is radii of curvature acceptable
			acceptable_radii = self.radius_of_curvature < params.ACCEPTABLE_RADII

			# if norm diff in poly coeff acceptable
			self.diffs = np.linalg.norm(self.best_fit - self.current_fit)
			acceptable_poly = self.diffs < params.ACCEPTABLE_POLY_DELTA

			if acceptable_radii and acceptable_poly:
				ret = True
		else:
			ret = False
		
		return ret

	def add_measurement(self, image, x, y):
		
		# get image
		self.image = image
		
		# add x, y pixels
		self.allx = x
		self.ally = y
		
		# calc current poly fit
		self.current_fit = search_around_best_fit()
	
	def approve_measurement(self):
		self.num_iter_approved += 1
		self.best_fit += self.current_fit/self.num_iter_approved

	def reject_measurement(self):
		self.num_iter_rejected += 1
	
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

		return fit
	
	def get_current_pixels_polyfit(self, x, y):
		return np.polyfit(y, x, 2)








