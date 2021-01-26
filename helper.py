import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

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
		gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
		img_size = (test_img.shape[1], test_img.shape[0])
		# Do camera calibration given object points and image points
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	else:
		mtx = np.eye(3, dtype=int)
		dist = np.zeros(3)

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

def unwarp(img, src, dst, plotVisual=False):
	#use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	#use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

	if plotVisual:
		fig, ax = plt.subplots(2, 2, figsize=(20,10))
		ax[0,0].imshow(img)
		ax[0,0].set_title('Original Image', fontsize=20)
		imshape = img.shape
		vertices = np.array([[(300,650),(620, 420), (650, 420), (1020, 650)]], dtype=np.int32)
		# vertices = np.array([[(100,imshape[0]),(440, 320), (520, 320), (880, imshape[0])]], dtype=np.int32)
		ax[0,1].imshow(region_of_interest(img, vertices), cmap='gray')
		ax[0,1].set_title('ROI', fontsize=20)
		ax[1,0].imshow(warped)
		ax[1,0].set_title('Warped', fontsize=20)
		ax[1,1].imshow(warped)
		ax[1,1].set_title('Warped', fontsize=20)
		plt.show()

	return warped, M