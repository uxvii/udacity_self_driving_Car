# **Finding Lane Lines on the Road** 


### Reflection

### 1. the steps of drawing  unique pipelines on the left and right side

1. read the image and convert it from rgb to hsv color space

2. select the v tunnel of the image, apply gaussian blur

3. apply canny algorithm to detect edges

4. apply "get_roi" function to get the region of interest
 
5. apply hough algorithm to find all the lines within the roi

6. filter all the lines with their slope. The remaining lines on the left are those whose slope between 0.5 and 2, the right side line ,between -0.5 and -2.

7. average the slope k and the y intercept b of lines on the right and left

8. draw the two lines with the averaged parameter of k and b 
    
	


### 2. potential shortcomings of my current pipeline


One potential shortcoming would be what would happen when the lane is kurved

Another shortcoming could be when the car does not run in the middle of the two pipelines


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to change the line-fitting to kurve-fitting using polynom

