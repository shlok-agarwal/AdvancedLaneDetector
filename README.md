## Advanced Lane Finding 

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
![image](https://user-images.githubusercontent.com/22652444/106409903-a6c2bf80-640f-11eb-8a7f-92e69d1d1a99.png)

* Apply a perspective transform to rectify binary image ("birds-eye view").
![image](https://user-images.githubusercontent.com/22652444/106409937-bf32da00-640f-11eb-8cfe-dc9401188ed4.png)

* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
![image](https://user-images.githubusercontent.com/22652444/106410055-fef9c180-640f-11eb-9af9-fe0316735fdf.png)

Important Files
---
* `main.py` is the main file to run the pipeline. In this file you can change the input image file or video file. The output for the image is displayed right away. For video files, you might want to specify a path here.
* `pipeline.py` is a class that acts as a pipeline to process the image frames, detect the lanes and plot them on the image. It also acts like a lane manager or coordinater for the left and right lane.
* `helper.py` is a file that contains handy functions for image processing. More specifically to detect lane lines
* `line.py` is a class that stores the information for the lane. Based on current and past detections, it finds the best fitting polynomial

To get started, run `main.py`.
