
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video_out_50s.mp4 "Video"



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained the ./src/calibration_modul.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

```
def undistortion(img):
	dist_pickle=pickle.load(open("wide_dist_pickle.p","rb"))
	mtx=dist_pickle["mtx"]
	dist=dist_pickle["dist"]

	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst

if __name__ == '__main__':

    img = cv2.imread('../test_images/test1.jpg')
    
    img_undistorted = undistortion(img)

    cv2.imwrite('../output_images/test_calibration_before.jpg', img)
    cv2.imwrite('../output_images/test_calibration_after.jpg', img_undistorted)

```



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at  `binary_modul.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

1. filter out white pixels using rgb_white
```python
def rgb_white(image):

	lower = np.array([100,100,200])
	upper = np.array([255, 255, 255])
	mask = cv2.inRange(image, lower, upper)
	rgb_w = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)#the white region in the original pic
	rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)#white to grey

	
	binary = np.zeros_like(rgb_w)
	binary[(rgb_w >= 20) & (rgb_w <= 255)] = 1 #in the grey img, the white pixel tend to be 255

	return binary #the map with either 0/1
```


2. filter out pixel using the s-channel of HSL space :
```python
def binary_s_channel(img):
	from global_var import hls_low as l
	from global_var import hls_high as h
	
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	s_channel=cv2.equalizeHist(s_channel)

	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= l) & (s_channel <= h)] = 1

	return s_binary #the map with either 0/1
```

3. filter out pixel using the sobel-x 
```python
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
```
4. combine them
```python
combined[((w==1)|(s==1)&(x==1))]=255


```
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in  in the file `example.py`  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
	h, w = img.shape[:2]

	src = np.float32([[w, h-10],    # br
			[0, h-10],    # bl
			[546, 460],   # tl
			[732, 460]])  # tr
	dst = np.float32([[w, h],       # br
			[0, h],       # bl
			[0, 0],       # tl
			[w, 0]])      # tr
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]




#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

the basic of the lane pixel detection is sliding windows

the code is in the file of `line_modul.py`  line 69-line197


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 204 through 235 in my code in `line_modul.py`

```python
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
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 238 through 265 in my code in `line_modul.py` in the function `draw_back()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out_50s.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

actually the programm fails when the road consists of two colors(half bright gray and half dark gray), then the pipeline on the right side will be not recognizeable.

to fix this problem, i used the white pixel mask to filter the white pixel of the right-side-pipeline, then the programm worked well

