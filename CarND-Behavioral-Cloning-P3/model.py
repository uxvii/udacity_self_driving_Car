


import csv
import cv2
import numpy as np

lines=[]
with open('./data/driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
	   lines.append(line)

images=[]
measurements=[]

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='./data/IMG/'+filename
        image=cv2.imread(current_path)
	######!!!!!!####bgr to rbg#######################################
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
###############################################################################
        images.append(image)
        if i==1:	
            measurement=float(line[3])+1.5
        elif i==2:
            measurement=float(line[3])-1.5
        else:
            measurement=float(line[3])
        measurements.append(measurement)
###data argument#################################
argumented_images,argumented_measurements=[],[]
for image,measurement in zip(images,measurements):
	argumented_images.append(image)
	argumented_measurements.append(measurement)
	argumented_images.append(cv2.flip(image,1))
	argumented_measurements.append(measurement*-1.0)

###################################
x_train=np.array(argumented_images)
y_train=np.array(argumented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#### pre-processing#####################
model= Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
###

####################################model 
# Build the model
batch = 128
epochs = 7
dropout_p = 0.75

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255) - .5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(dropout_p))
model.add(Flatten())
model.add(Dense(1162, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
#############################################################
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=batch,validation_split=0.2,shuffle=True)
#############################################################
model.save('model.h5')
exit()




















