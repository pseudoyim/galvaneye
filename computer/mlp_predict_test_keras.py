__author__ = 'zhengwang & paulyim'

import cv2
import numpy as np
import glob
import keras


# load training data
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('testing_data/*.npz')

for single_npz in training_data:                    #''' Isn't this the same data we used to train the model?  Should we have split the data at some point before? '''
    with np.load(single_npz) as data:
        print data.files
        test_temp = data['train']
        test_labels_temp = data['train_labels']
        print test_temp.shape
        print test_labels_temp.shape
    image_array = np.vstack((image_array, test_temp))
    label_array = np.vstack((label_array, test_labels_temp))

test = image_array[1:, :]
test_labels = label_array[1:, :]
print test.shape
print test_labels.shape

# Load the saved neural network model
model = keras.models.load_model('nn_h5/nn.h5')

e0 = cv2.getTickCount()

print 'Evaluating model performance on test set...'
# Evalute trained model on TEST set
score = model.evaluate(test, test_labels, batch_size=1000)
loss = score[0]
accuracy = score[1]

e00 = cv2.getTickCount()

time0 = (e00 - e0)/cv2.getTickFrequency()

print ''
print 'Time taken to evaluate model: ', time0, 'seconds'
print 'Loss score: ', loss
print 'Accuracy score: ', accuracy
