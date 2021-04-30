## Advanced Lane Finding 

![](project-gif.gif)

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

Discussion and Future Work
---
This project is a combination of image processing algorithms to segment the lane pixels and polynomial fitting to fit lane lines. A sliding window search is used to find the lane lines. To save on processing time and knowing that the lane lines do not change much after every frame, a search is conducted around the polynomial to find the new coefficients every frame. A heuristic approach determines if we need to run the sliding window search again.

CV projects often involve very fine tuning so this algorithm and parameters selected might not be suitable for real world scenario yet. A better heuristic approach to determine the quality of lane detection would be extremely helpful to filter outliers and get more accurate lane line. Better image thresholding and perspective transform thresholds would help in different lighting conditions and when the car is not aligned in the center of the lane.

