

**Build a Traffic Sign Recognition Project**



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogramm showing how the data distribute

![avatar][https://github.com/uxvii/udacity_self_driving_Car/blob/master/CarND-Traffic-Sign-Classifier-P2/histogramm_test_Set.png.png]
![avatar][https://github.com/uxvii/udacity_self_driving_Car/blob/master/CarND-Traffic-Sign-Classifier-P2/histogramm_training_Set.png]
![avatar][https://github.com/uxvii/udacity_self_driving_Car/blob/master/CarND-Traffic-Sign-Classifier-P2/histogramm_validation_Set.png.png]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

 I normalized the image data because numerical error can be avoided.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|										
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 | 
| RELU					|										
| Max pooling	      	| 2x2 stride,  outputs 5x5x16|				
| Fully connected	| Input = 400. Output = 120.|
|RELU|
|dropot(0.75)|
| Fully connected	| Input = 120. Output = 84|
|RELU|
|dropot(0.75)|       									
| Fully connected	|input =84   output=43 		|								

#### 3. Describe how you trained your model. 
optimizer:AdamOptimizer
batch size: 200
number of epochs:100
learning rate:0.01
dropout prob=0.75

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.97
* test set accuracy of 0.949

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  -lenet architecture, because its easy 

* What were some problems with the initial architecture?
  -the accuracy is low both in the test_Set and the validation set

* How was the architecture adjusted and why was it adjusted? 
  -due to the underfitting, i changed the output of the first conv layer from depth 6 to depth 16
  - to prevent overfitting , i added two dropout layers
	
* Which parameters were tuned? How were they adjusted and why
  - i tuned the dropout prob.

* What are some of the important design choices and why were they chosen? 
  - the drop layer is important because it prevents overfittng
  - the relu layer is important because it add non linearity



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][1.jpeg] ![alt text][3.jpeg] ![alt text][9.jpeg] 
![alt text][14.jpeg] ![alt text][22.jpeg]

The 1st 3rd image might be difficult to classify because the area of the sign is small compared to the whole area of the whole pic. And the background of the pictures may be different from the pics in the training set. this may maybe the reason why they are not correctly predictet

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| label=14      		| label=14  									| 
| label=1     			| label=0 |
| label=3			|label=38 	|									
| label=9	      		| label=18 	|
| label=22		        | label=28      	   	|
| label=38	                | label=13    	   	   |

The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 16.7%. this is quite bad prediction

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|1st picture   |label 14   |
|predicted label  |14, 15,  5, 25,  2|
|probabilities |  0.99,   5.e-04,   2.3e-05, 4.6e-06, 9.5e-08|

|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|2nd picture   |label 1   |
|predicted label  |  0, 17,  1,  3, 20|
|probabilities| 0.9,   2.05e-02,   3.9e-03, 1.1e-04,   9.7e-05|
|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|3rd picture   |label 3   |
|predicted label  | 38, 10,  9, 12, 40|
|probabilities|  0.6,   0.2,   0.15, 1.12e-03,   9.7e-04|
|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|4th picture   |label 9   |
|predicted label  | 18, 26, 12, 11,  1|
|probabilities|0.9,   5.32e-03,   1.04e-03, 2.4e-05,   2.8e-07|
|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|5th picture   |label 22  | 
|predicted label  | 28, 29, 22,  8,  3|
|probabilities| 0.9,   9.34e-02,   1.7e-09, 1.39e-09,   8.1e-13|

|      		|    	        					| 
|:---------------------:|:---------------------------------------------:| 
|6th picture   |label 38 | 
|predicted label  | 13, 38, 34, 30, 33|
|probabilities|0.9,   3.07e-03,   9.8e-06, 3.4e-08,   6.6e-12|

