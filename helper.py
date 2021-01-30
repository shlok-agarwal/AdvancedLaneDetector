import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle as pickle

def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	(assuming your grayscaled image is called 'gray')
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Or use BGR2GRAY if you read an image with cv2.imread()
	# return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	`vertices` should be a numpy array of integer points.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.    
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, lane_center_frac):
	"""
	`img` should be the output of a Canny transform.
		
	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines, int(lane_center_frac*img.shape[1]))
	return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.
	
	`initial_img` should be the image before any processing.
	
	The result image is computed as follows:
	
	initial_img * α + img * β + γ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, γ)


# need to run this once to get the intrinsic matrix
def get_cal_mtx(test_img, nx = 8, ny = 6, fname_dir = 'camera_cal/calibration*.jpg'):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	# Make a list of calibration images
	images = glob.glob(fname_dir)

	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

	# if any points found
	if len(objpoints) > 0:
		# Do camera calibration given object points and image points
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_img.shape[1:], None, None)
	else:
		mtx = np.eye(3, dtype=int)
		dist = np.zeros(3)
	
	print(dist)
	print(mtx)
	data = {'mtx': mtx, 'dist': dist}
	pickle.dump(data, open("camera_calibration.p", "wb"))

	return mtx, dist

def cal_undistort(img, mtx, dist): 
	return cv2.undistort(img, mtx, dist, None, mtx)

def applySobelOperator(image, plotVisual = False):

	def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100)):
		
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# 2) Take the derivative in x or y given orient = 'x' or 'y'
		if orient == 'x':
			sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		else: 
			sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# 3) Take the absolute value of the derivative or gradient
		abs_sobel = np.absolute(sobel)

		# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# 5) Create a mask of 1's where the scaled gradient magnitude 
				# is > thresh_min and < thresh_max
		sbinary = np.zeros_like(scaled_sobel)
		sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return sbinary

	def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# 3) Calculate the magnitude 
		abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
		# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
		scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
		# 5) Create a binary mask where mag thresholds are met
		sbinary = np.zeros_like(scaled_sobel)
		sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return sbinary

	def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
		
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# 3) Take the absolute value of the x and y gradients
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
		orientation = np.arctan2(abs_sobely, abs_sobelx)
		# 5) Create a binary mask where direction thresholds are met
		sbinary = np.zeros_like(orientation)
		sbinary[(orientation >= thresh[0]) & (orientation <= thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return sbinary
	
	# Choose a Sobel kernel size
	ksize = 3 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
	grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
	mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
	dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
	combined = np.zeros_like(dir_binary)
	combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	if plotVisual:
		# Visualize undistortion
		fig, ax = plt.subplots(2, 3, figsize=(20,10))
		ax[0,0].imshow(image)
		ax[0,0].set_title('Original Image', fontsize=20)
		ax[0,1].imshow(gradx, cmap='gray')
		ax[0,1].set_title('Sobel Grad X', fontsize=20)
		ax[0,2].imshow(grady, cmap='gray')
		ax[0,2].set_title('Sobel Grad Y', fontsize=20)
		ax[1,0].imshow(mag_binary, cmap='gray')
		ax[1,0].set_title('Sobel Mag', fontsize=20)
		ax[1,1].imshow(dir_binary, cmap='gray')
		ax[1,1].set_title('Sobel Dir', fontsize=20)
		ax[1,2].imshow(combined, cmap='gray')
		ax[1,2].set_title('Combined', fontsize=20)
		plt.show()
	
	return combined

def applyHSVAndSobelXFilter(img, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100), plotVisual = False):
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	
	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
   
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

	if plotVisual:
		# Visualize undistortion
		fig, ax = plt.subplots(2, 2, figsize=(20,10))
		ax[0,0].imshow(img)
		ax[0,0].set_title('Original Image', fontsize=20)
		ax[0,1].imshow(sxbinary, cmap='gray')
		ax[0,1].set_title('Sobel Grad X', fontsize=20)
		ax[1,0].imshow(s_binary, cmap='gray')
		ax[1,0].set_title('HSV filter', fontsize=20)
		ax[1,1].imshow(combined_binary, cmap='gray')
		ax[1,1].set_title('Combined', fontsize=20)
		plt.show()

	return combined_binary

def warp(img, src, dst, plotVisual=False):
	#use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	#use cv2.warpPerspective() to warp your image to a top-down view
	img_shape = (img.shape[1], img.shape[0])
	warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_LINEAR)

	if plotVisual:
		fig, ax = plt.subplots(1, 2, figsize=(20,10))
		ax[0].set_title('Original Image', fontsize=20)
		cv2.polylines(img, [src.astype(np.int32)],True, (1,100,100), thickness=2)
		ax[0].imshow(img, cmap='gray')
		ax[0].plot(src[0][0], src[0][1], 'r+')
		ax[0].plot(src[1][0], src[1][1], 'c^')
		ax[0].plot(src[2][0], src[2][1], 'r^')
		ax[0].plot(src[3][0], src[3][1], 'g^')
		ax[1].imshow(warped,  cmap='gray')
		ax[1].set_title('Warped', fontsize=20)
		plt.show()

	return warped, M

def find_lane_pixels(binary_warped):
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nwindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated later for each window in nwindows
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		### TO-DO: Find the four below boundaries of the window ###
		win_xleft_low = leftx_current - margin  
		win_xleft_high = leftx_current + margin  
		win_xright_low = rightx_current - margin  
		win_xright_high = rightx_current + margin  
		
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),
		(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),
		(win_xright_high,win_y_high),(0,255,0), 2) 
		
		### TO-DO: Identify the nonzero pixels in x and y within the window ###
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		### TO-DO: If you found > minpix pixels, recenter next window ###
		### (`right` or `leftx_current`) on their mean position ###
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		
		if len(good_right_inds) > minpix:
			rightx_current =  np.int(np.mean(nonzerox[good_right_inds]))
	
	# Concatenate the arrays of indices (previously was a list of lists of pixels)
	try:
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
		# Avoids an error if the above is not implemented fully
		pass

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	return leftx, lefty, rightx, righty, out_img


def fit_polynomial_from_lane_pixels(binary_warped):
	# Find our lane pixels first
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

	### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
	# calculate polynomial

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	return left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit, plotVisual=False):
	
	def fit_poly(img_shape, leftx, lefty, rightx, righty):
		### Fit a second order polynomial to each with np.polyfit() ###     
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		# Generate x and y values for plotting
		ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
		###  Calc both polynomials using ploty, left_fit and right_fit ###
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		
		return left_fitx, right_fitx, ploty
	
	# HYPERPARAMETER
	# Choose the width of the margin around the previous polynomial to search
	# The quiz grader expects 100 here, but feel free to tune on your own!
	margin = 100

	# Grab activated pixels
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	### TO-DO: Set the area of search based on activated x-values ###
	### within the +/- margin of our polynomial function ###
	### Hint: consider the window areas for the similarly named variables ###
	### in the previous quiz, but change the windows to our new search area ###
	
	left_lane_inds = []
	right_lane_inds =[]
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
					left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
					left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
					right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
					right_fit[1]*nonzeroy + right_fit[2] + margin)))
 
	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit new polynomials
	left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
	
	## Visualization ##
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
							  ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
							  ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	
	if plotVisual:
		# Plot the polynomial lines onto the image
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.imshow(result)
		plt.show()
		## End visualization steps ##
	
	return result

def plot_lanes(image, left_poly, right_poly):
	
	margin = 100
	
	## Visualization ##
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((image, image, image))*255
	window_img = np.zeros_like(out_img)
	
	# Generate x and y values for plotting
	img_shape = image.shape
	ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
	### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
	left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
	right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
							ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
							ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	
	## End visualization steps ##
	return result

