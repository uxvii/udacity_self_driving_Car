import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from calibration_modul import undistortion
import matplotlib.image as mpimg

def perspective_transform(img):
	'''
	src = np.float32([[586.,459.],
		          [698.,459.],
		          [1038.,681.],
		          [286.,681.]])

        
	h,w = img.shape[:2]

	dst = np.float32([[200,200],[w-200,200],[w-200,h],[200,h]])
       '''

	h, w = img.shape[:2]

	src = np.float32([[w, h-10],    # br
			[0, h-10],    # bl
			[546, 460],   # tl
			[732, 460]])  # tr
	dst = np.float32([[w, h],       # br
			[0, h],       # bl
			[0, 0],       # tl
			[w, 0]])      # tr


	M = cv2.getPerspectiveTransform(src, dst)
	Minv=cv2.getPerspectiveTransform(dst,src)

	warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
	
	return warped, M, Minv


if __name__ == '__main__':


	img = cv2.imread('../test_images/straight_lines2.jpg')
    
	img_undistorted = undistortion(img)

	warped,M,Minv=perspective_transform(img_undistorted)


	cv2.imwrite('../output_images/bird_eye.jpg', warped)
