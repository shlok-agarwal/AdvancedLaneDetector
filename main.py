from pipeline import *
from moviepy.editor import VideoFileClip


# TEST FILES
image_fname = 'test_images/straight_lines1.jpg'
# image_fname = 'test_images/straight_lines2.jpg'
# image_fname = 'test_images/test1.jpg' 
# image_fname = 'test_images/test2.jpg'
# image_fname = 'test_images/test3.jpg'
# image_fname = 'test_images/test4.jpg'
# image_fname = 'test_images/test5.jpg'
# image_fname = 'test_images/test6.jpg'

video_fname_op = 'project_output.mp4'
video_fname_ip = "project_video.mp4"

# video_fname_op = 'challenge_video_op.mp4'
# video_fname_ip = "challenge_video.mp4"

## Create global variables
pipeline = Pipeline()

test = 'IMAGE'

if test == 'IMAGE':
	# Test functions
	image = mpimg.imread(image_fname)
	pipeline.process_image(image, False, True)
else:
	video = VideoFileClip(video_fname_ip)
	clip = video.fl_image(lambda img: pipeline.process_image(img, False, False))
	clip.write_videofile(video_fname_op, audio=False)
	

