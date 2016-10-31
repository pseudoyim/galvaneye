import csv
import glob
import json
import numpy as np
import pandas as pd
import time

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


print 'Loading training data...'
time_load_start = time.time()         # Returns the number of ticks after a certain event.

# Load training data, unpacking what's in the saved .npz files.
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('training_data/*.npz')        # Finds filename matching specified path or pattern.

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

# # Normalize with l2 (not gonna use this...)
# X = preprocessing.normalize(X, norm='l2')

# Normalize from 0 to 1
X = X / 255.

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = Sequential()

time_load_end = time.time()
time_load_total = time_load_end - time_load_start
print 'Total time taken to load image data:', time_load_total, 'seconds'


# Get start time of Training
time_training_start = time.time()

print 'Training...'
# Dense(n) is a fully-connected layer with n hidden units in the first layer.
# You must specify the expected input data shape (e.g. input_dim=20 for 20-dimensional input vector).
model.add(Dense(30, input_dim=38400, init='uniform'))
model.add(Dropout(0.2))
model.add(Activation('relu'))
# model.add(Dense(10, init='uniform' ))
# model.add(Dropout(0.2))
# model.add(Activation('softmax'))
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
model.save('nn_h5/nn_{}.h5'.format(str(training_data)[-33:-6]))

# Save parameters to json file
json_string = model.to_json()
with open('./logs/nn_params_json/nn_{}'.format(str(training_data)[-33:-6]), 'w') as new_json:
    json.dump(json_string, new_json)

# Save training results
row_params = [training_data[-31:], loss, accuracy]
with open('./log_nn_training.csv','a') as log:
    log_writer = csv.writer(log)
    log_writer.writerow(row_params)
