
from binary_modul import binary_combined
import numpy as np
import cv2

import collections
import matplotlib.pyplot as plt

from global_var import time_window,ym_per_pix,xm_per_pix
from calibration_modul import undistortion

import matplotlib.image as mpimg
import pickle
from perspective_transform import perspective_transform
from global_var import nwindows
import os
from moviepy.editor import VideoFileClip
from line_modul import Line_Detecter


def process_image(img):
	ll= Line_Detecter(time_window)


	img_undistorted = undistortion(img)

	img_binary = binary_combined(img_undistorted)



	warped,M,Minv=perspective_transform(img_binary)


	result_img=ll.Detect(warped,img_undistorted,Minv)

	return result_img

output = '../project_video_out_50s.mp4'
clip1 = VideoFileClip("../project_video.mp4")#.subclip(40,43)
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)
