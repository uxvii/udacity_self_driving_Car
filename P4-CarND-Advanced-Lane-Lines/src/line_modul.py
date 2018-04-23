
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








class Line_Detecter:

	def __init__(self, time_window):


        # polynomial coefficients fitted on the last iteration
		self.last_ABC_left = None
		self.last_ABC_right = None

		self.last_ABC_left_meter = None
		self.last_ABC_right_meter = None

        # list of polynomial coefficients of the last N iterations
		self.ten_ABC_left = collections.deque(maxlen=time_window)
		self.ten_ABC_right = collections.deque(maxlen=time_window)

		self.ten_ABC_left_meter = collections.deque(maxlen=time_window)
		self.ten_ABC_right_meter = collections.deque(maxlen=time_window)


		self.radius_of_curvature = None
	#if there is a base line     True     / or no base    False
		self.base=False

		self.distance=None

		self.leftx_ = None
		self.lefty_ = None
		self.rightx_ = None
		self.righty_ = None

		
####################################################methods##################################
	#add new ABC into the 10ABC
	def update_line(self):

		self.ten_ABC_left.append(self.last_ABC_left)
		self.ten_ABC_right.append(self.last_ABC_right)
		
		self.ten_ABC_left_meter.append(self.last_ABC_left_meter)
		self.ten_ABC_right_meter.append(self.last_ABC_right_meter)


#############if base ==False

	def sliding_window(self,binary_warped):
		
		#Add all the values of the under half picutre 
		histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)


		

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		
		# Set height of windows
		window_height = np.int(binary_warped.shape[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []
		'''
		play with the index
		'''
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			'''
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			'''
			# Identify the nonzero pixels in x and y within the window

			'''
return the same dimension of nonzeroy      (10011101011....)  from the & operation.  
get the nonzero position(==lane index of nonzero)   (1,2,5,7,8.......)
			'''
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)   ##merge the list 
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		'''
		extraction list[list2]
		'''
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit ABC
		self.last_ABC_left  = np.polyfit(lefty, leftx, 2)
		self.last_ABC_right = np.polyfit(righty, rightx, 2)
		#abc real
		self.last_ABC_left_meter=np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
		self.last_ABC_right_meter=np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


		self.update_line()
		

		self.leftx_ = leftx
		self.lefty_ = lefty
		self.rightx_ = rightx
		self.righty_ = righty



		return 0
			 





#########if Base==True
	def find_line_with_base(self,binary_warped):
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100

		left_lane_inds = ((nonzerox > (self.last_ABC_left[0]*(nonzeroy**2) + self.last_ABC_left[1]*nonzeroy + self.last_ABC_left[2] - margin)) & (nonzerox < (self.last_ABC_left[0]*(nonzeroy**2) + self.last_ABC_left[1]*nonzeroy + self.last_ABC_left[2] + margin))) 

		right_lane_inds = ((nonzerox > (self.last_ABC_right[0]*(nonzeroy**2) + self.last_ABC_right[1]*nonzeroy + self.last_ABC_right[2] - margin)) & (nonzerox < (self.last_ABC_right[0]*(nonzeroy**2) + self.last_ABC_right[1]*nonzeroy + self.last_ABC_right[2] + margin)))  

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		self.last_ABC_left = np.polyfit(lefty, leftx, 2)
		self.last_ABC_right = np.polyfit(righty, rightx, 2)

		self.last_ABC_left_meter=np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
		self.last_ABC_right_meter=np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
		
		self.update_line()


		return 0


##get the curvatur radius from the ten abc
	def Get_Radius_Meter(self):

		y_meter=700*ym_per_pix

		left_fit_cr = np.mean(self.ten_ABC_left_meter, axis=0)
		right_fit_cr = np.mean(self.ten_ABC_right_meter, axis=0)

		# Calculate the new radii of curvature
		left_radius = ((1 + (2*left_fit_cr[0]*(y_meter) + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_radius = ((1 + (2*right_fit_cr[0]*(y_meter) + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

		self.radius_of_curvature=(left_radius+right_radius)/2

		return 0

	def Get_Position(self):

		ploty=700.

		left_fit = np.mean(self.ten_ABC_left, axis=0)
		right_fit = np.mean(self.ten_ABC_right, axis=0)


		left_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		lane_middle=0.5*(left_x+right_x)
		
		pixel_distance=np.absolute(lane_middle-1280/2)
		self.distance=pixel_distance*xm_per_pix

		return 0

		
	def draw_back(self,binary_warped,Minv):

		left_fit = np.mean(self.ten_ABC_left, axis=0)
		right_fit = np.mean(self.ten_ABC_right, axis=0)
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		line_pts = np.hstack((line_window1, line_window2))


		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([line_pts]), (0,255,0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(window_img, Minv, (binary_warped.shape[1], binary_warped.shape[0]))

		
		return newwarp##(3 layer)


	
		


		
	def Detect(self,binary_warped,img,Minv):
		if self.base==False:
			self.sliding_window(binary_warped)
			self.base=True
		else:
			self.find_line_with_base(binary_warped)

		self.Get_Radius_Meter()
		self.Get_Position()
		
		mask=self.draw_back(binary_warped,Minv)


		result = cv2.addWeighted(img, 1, mask, 0.3, 0)
		#print radius and position infomation 
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(result, 'Curvature radius: {:.02f}m'.format(self.radius_of_curvature), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(result, 'Offset from center: {:.02f}m'.format(self.distance), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

		return result



def show(img):
	
	plt.imshow(img)
	plt.show()
	return 0




if __name__ == '__main__':

	ll= Line_Detecter(time_window)


	# show result on test images
	img=mpimg.imread('../fail/6.jpg')

	
	img_undistorted = undistortion(img)

	img_binary = binary_combined(img_undistorted)




	warped,M,Minv=perspective_transform(img_binary)

	#####################################################
	result_img=ll.Detect(warped,img_undistorted,Minv)
	show(result_img)

	mpimg.imsave('../output_images/test_line.jpg',result_img)
	











