import csv
import cv2
import glob
import json
import numpy as np
import pandas as pd
import time

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from tqdm import tqdm

print 'Loading training data...'
time_load_start = time.time()         # Returns the number of ticks after a certain event.

# Load jpg images to get image_array.
training_images = glob.glob('./training_data/*_cnn/*.jpg')
image_array = np.array([cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in tqdm(training_images)], dtype=np.float64)

# Load training data .npz to get label_array, unpacking what's in the saved .npz files.
# label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('training_data/*.npz')        # Finds filename matching specified path or pattern.

label_array = None
for single_npz in training_data:                        # single_npz == one array representing one array of saved image data and user input label for that image.
    with np.load(single_npz) as data:
        train_labels_temp = data['train_labels']        # returns the training user input data array assigned to 'train_labels' argument created during np.savez step in 'collect_training_data.py'
        print 'Original labels shape:', train_labels_temp.shape
    label_array = np.array([label for label in tqdm(train_labels_temp)], dtype=np.float64)

X = image_array
y = label_array
print 'Shape of feature array: ', X.shape
print type(X)
print 'Shape of label array: ', y.shape
print type(y)

# # Normalize with l2 (not gonna use this...)
# X = preprocessing.normalize(X, norm='l2')

# Normalize from 0 to 1
X = X / 255.
# for each in tqdm(X):
#     each/255.

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape


model = Sequential()

time_load_end = time.time()
time_load_total = time_load_end - time_load_start
print 'Total time taken to load image data:', time_load_total, 'seconds'


# Get start time of Training
time_training_start = time.time()

print 'Training...'


# CONVOLUTION

# When using THEANO backend
model.add(Reshape((1, 120, 320), input_shape=(120, 320)))

# # When using TENSORFLOW backend
# model.add(Reshape((120, 320, 1), input_shape=(120, 320)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(30, init='uniform'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(3, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=1000,
          validation_data=(X_test, y_test))

print 'outputshape:', model.outputshape


# Get end time of Training
time_training_end = time.time()
time_training_total = time_training_end - time_training_start
print ''
print 'Total time taken to train model:', time_training_total, 'seconds'

# Evalute trained model on TEST set
print ''
print 'Evaluation of model on test holdout set:'
score = model.evaluate(X_test, y_test, batch_size=1000)
loss = score[0]
accuracy = score[1]
print ''
print 'Loss score: ', loss
print 'Accuracy score: ', accuracy

# Save model as h5
timestr = time.strftime('%Y%m%d_%H%M%S')
filename_timestr = 'nn_{}.h5'.format(timestr)
model.save('nn_h5/nn_{}.h5'.format(timestr))

# Save parameters to json file
json_string = model.to_json()
with open('./logs/nn_params_json/nn_{}.json'.format(timestr), 'w') as new_json:
    json.dump(json_string, new_json)

# Save training results to csv log
row_params = [str(training_data)[-33:-2], filename_timestr, loss, accuracy]
with open('./logs/log_nn_training.csv','a') as log:
    log_writer = csv.writer(log)
    log_writer.writerow(row_params)
