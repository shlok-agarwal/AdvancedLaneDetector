from helper import*

#### CALIBRATION ####

#reading in test image
test_calib_img = mpimg.imread('camera_cal/calibration3.jpg')

# get camera matrix
mtx, dist = get_cal_mtx(test_calib_img, 9, 6)