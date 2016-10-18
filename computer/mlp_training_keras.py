__author__ = 'zhengwang & paulyim'

import cv2
import numpy as np
import pandas as pd
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn import preprocessing


print 'Loading training data...'
e0 = cv2.getTickCount()         # Returns the number of ticks after a certain event.

# Load training data, unpacking what's in the saved .npz files.
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('training_data/*.npz')        # Finds filenames matching specified path or pattern.

for single_npz in training_data:                        # single_npz == one array representing one array of saved image data and user input label for that image.
    with np.load(single_npz) as data:
        print data.files
        train_temp = data['train']                      # returns the training data image array assigned to 'train' argument created during np.savez step in 'collect_training_data.py'
        train_labels_temp = data['train_labels']        # returns the training user input data array assigned to 'train_labels' argument created during np.savez step in 'collect_training_data.py'
        print train_temp.shape
        print train_labels_temp.shape
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :]
y = label_array[1:, :]
print 'Shape of feature array: ', X.shape
print 'Shape of label array: ', y.shape

# Normalize all columns in X. (Is this necessary in this context?)
X_normalized = preprocessing.normalize(X, norm='l2')
# features = df.columns
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.20)
model = Sequential()

e00 = cv2.getTickCount()
time0 = (e00 - e0)/ cv2.getTickFrequency()      # Returns the number of ticks per second.
print 'Total time taken to load image data:', time0, 'seconds'

# Get start time of Training
e1 = cv2.getTickCount()



print 'Training...'
# Dense(n) is a fully-connected layer with n hidden units in the first layer.
# You must specify the expected input data shape (e.g. input_dim=20 for 20-dimensional input vector).
model.add(Dense(32, input_dim=38400, init='uniform'))
model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Dense(15, init='uniform' ))
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(3, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train,
          nb_epoch=50,
          batch_size=1000)


# Get end time of Training
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print ''
print 'Total time taken to train model:', time, 'seconds'


# Evalute trained model on TEST set
score = model.evaluate(X_test, y_test, batch_size=1000)

loss = score[0]
accuracy = score[1]

print ''
print 'Loss score: ', loss
print 'Accuracy score: ', accuracy


# Save model as h5 (or should I just save the weights?)
model.save('nn_h5/nn.h5')
