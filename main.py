from pipeline import *
from moviepy.editor import VideoFileClip

image_fname = 'test_images/test6.jpg'
video_fname_op = 'project_output.mp4'
video_fname_ip = "project_video.mp4"

## Create global variables
pipeline = Pipeline()

test = 'IMAGE'

if test == 'IMAGE':
	# Test functions
	image = mpimg.imread(image_fname)
	pipeline.process_image(image, True)
else:
	video = VideoFileClip(video_fname_ip)
	clip = video.fl_image(lane_process_pipeline)
	clip.write_videofile(video_fname_op, audio=False)
	

