

**Behavioral Cloning Project**

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 62-80) 

The model includes RELU layers to introduce nonlinearity (code line 65), and the data is normalized in the model using a Keras lambda layer (code line 52). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 66). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

And more importently, the images from the Left and Right camera were a great help to turn the car from the side to the middle

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet.

But unfortunatly, the car was not able to run on the track.

Then I increased the Complexicity of the Network
I added several Conv2d layers and Dropout layers and fully connected layers


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-80) consisted of a convolution neural network with the following layers and layer sizes :

Conv2d     depth 24       kernal_size 5x5
dropout
Conv2d     depth 36       kernal_size 5x5
dropout
Conv2d     depth 48       kernal_size 5x5
dropout
Conv2d     depth 64       kernal_size 3x3
dropout
fully connected layer 1162
fully connected layer 100
fully connected layer 50
fully connected layer 10
fully connected layer 1
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving. A


To augment the data sat, I also flipped images and angles thinking that this would help the car to turn back to the middle when its on the right/left side.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
