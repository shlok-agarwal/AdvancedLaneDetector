import numpy as np

class Line():

    def __init__(self):
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
        # number of iteration
        self.num_iter = 0

    # Use this function to completely reset the line parameters and start from scratch 
    def reset(self):
        self.num_iter = 0

    def add_measurement(self, x, y):
        self.allx = x
        self.ally = y
        self.num_iter += 1

        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.best_fit += self.current_fit/self.num_iter


