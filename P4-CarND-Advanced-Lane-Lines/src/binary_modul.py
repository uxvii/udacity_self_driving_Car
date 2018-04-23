import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# rgb thresholding for white (best)
def rgb_white(image):

	lower = np.array([100,100,200])
	upper = np.array([255, 255, 255])
	mask = cv2.inRange(image, lower, upper)
	rgb_w = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)#the white region in the original pic
	rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)#white to grey

	
	binary = np.zeros_like(rgb_w)
	binary[(rgb_w >= 20) & (rgb_w <= 255)] = 1 #in the grey img, the white pixel tend to be 255

	return binary #the map with either 0/1
	
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
	w=rgb_white(img)
	s=binary_s_channel(img)
	x=binary_gradient(img)
	combined=np.zeros_like(s)
	combined[((w==1)|(s==1)&(x==1))]=255

	#cv2.imwrite('../output_images/binary_result.jpg', combined)
	return combined


def show(img):
	
	plt.imshow(img)
	plt.show()
	return 0



if __name__ == '__main__':
	#img=mpimg.imread("../fail/t8.png")   ../test_images/test2.jpg      ../fail/t8.png
	img=cv2.imread("../test_images/test6.jpg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	show(img)
	out=binary_combined(img)
	show(out)
	

	white_binary=rgb_white(img)
	show(white_binary)
	s_binary=binary_s_channel(img)
	show(s_binary)
	sxbinary=binary_gradient(img)
	show(sxbinary)
	
	'''
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	eq = cv2.equalizeHist(s_channel)
	show(eq)
	'''


