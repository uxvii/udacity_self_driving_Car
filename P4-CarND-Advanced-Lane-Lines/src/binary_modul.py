import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


#hue mask
def hue_mask(img):
	from global_var import hue_low as l
	from global_var import hue_high as h
	
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	h_channel = hls[:,:,0]

	h_binary = np.zeros_like(h_channel)
	h_binary[(h_channel >= l) & (h_channel <= h)] = 1

	return h_binary #the map with either 0/1
	
#hls   s  channel
def binary_s_channel(img):
	from global_var import hls_low as l
	from global_var import hls_high as h
	
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	s_channel=cv2.equalizeHist(s_channel)

	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= l) & (s_channel <= h)] = 1

	return s_binary #the map with either 0/1



#  magnitute of sobel x

def binary_gradient(img):
	from global_var import sobel_size,sobel_x_low,sobel_x_high

	#gray equalizing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eq = cv2.equalizeHist(gray)
	
	sobelx = cv2.Sobel(eq, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sobel_x_low) & (scaled_sobel <= sobel_x_high)] = 1

	return sxbinary

	
def binary_combined(img):
	h_binary=hue_mask(img)
	s_binary=binary_s_channel(img)
	sxbinary=binary_gradient(img)
	combined=np.zeros_like(s_binary)
	combined[((s_binary==1)|(sxbinary==1))&(h_binary==1)]=255

	#cv2.imwrite('../output_images/binary_result.jpg', combined)
	return combined

if __name__ == '__main__':
	img=mpimg.imread('../test_images/straight_lines2.jpg')
	binary_combined(img)	
	

'''
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	eq = cv2.equalizeHist(s_channel)
	cv2.imwrite('../output_images/s_channel.jpg', eq)
'''		


