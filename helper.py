import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle as pickle

'''
This file contains handy functions for image processing. More specifically to detect lane lines
'''

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

# undistort the image
def cal_undistort(img, mtx, dist): 
	return cv2.undistort(img, mtx, dist, None, mtx)

# apply HSV and soble filter on image
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
	# combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	combined_binary = cv2.addWeighted(s_binary, 1, sxbinary, 0.6, 0)

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

# birds eye view by wrapping image
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

# find lane pixels using sliding window search
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

# given the left and right lane polynomials, plot a lane on the image
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

	# # Generate a polygon to illustrate the search window area
	# # And recast the x and y points into usable format for cv2.fillPoly()
	# left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	# left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	# 						ploty])))])
	# left_line_pts = np.hstack((left_line_window1, left_line_window2))
	# right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	# right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	# 						ploty])))])
	# right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# # Draw the lane onto the warped blank image
	# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	# result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	result = cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

	## End visualization steps ##
	return result

def annotate(img, r1, r2):
	rows, cols = img.shape[0], img.shape[1]
	font = cv2.FONT_HERSHEY_SIMPLEX
	dst = img
	pos = 0.12
	cv2.putText(dst,'Left curve radius = {:.0f}m'.format(r1), (np.int(cols/2)-100,100), font, 1,(255,255,255),2)
	cv2.putText(dst,'Right curve radius = {:.0f}m'.format(r2), (np.int(cols/2)-100,150), font, 1,(255,255,255),2)
	cv2.putText(dst,'Position = {:1.2}m left'.format(pos), (np.int(cols/2)-100,50), font, 1,(255,255,255),2)
	return dst

