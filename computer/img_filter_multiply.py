'''
This module does the following:
  1. Reads original image files saved in a directory.
  2. Applies a filter to each.
  3. Decodes the filtered image into a numpy array.
  4. Appends the label (direction inputted by driver) to each array.
  5. Copies the above new array and flips the image and label (if left then right, and vice versa).
  6. Saves the resulting arrays to a npz file, for consumption in neural net training.
'''

import numpy as np
import cv2
import glob



class ImageFilterMultiplier(object):

    def __init__(self, originals, filter_option):

        # Variables for each preprocessing/filter option.
        # optionA =


    def apply_filter(self):
        # image_array and label_array each start off as one row of zeros. As images are added, they are vstacked from below.
        image_array = np.zeros((1, 38400))      # Image resolution is 320x240. But we are lopping off the top half, so actually 320x120. 320 * 120 = 38400.

        # original images, read them in one-by-one
        originals = glob.glob('training_images/*.jpg')

        for each in originals:                        # single_npz == one npz file in the 'training_data' folder.
            with cv2.read(each) as image:
                # Decode/reshape image into np.array
                temp_array = image.reshape(1, 38400).astype(np.float32)
                image_array = np.vstack((image_array, train_temp))





            label_array = np.vstack((label_array, train_labels_temp))

        # START HERE
        # After the image_array is done (every image has been decoded and vstacked),
        # can you just unpack the npz file for labels and concatenate it to the image_array?



        # overlay = image.copy()  IF putText works below, then delete this line. Otherwise you'll need to add a transparent overlay (cv2.addWeighted(overlay...)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # images with Canny filter applied:
        wide = cv2.Canny(blurred, 10, 200)
        tight = cv2.Canny(blurred, 225, 250)
        auto = self.auto_canny(blurred)

        # select lower half of the image. 0:120 would be the upper half of rows. 120:240 is the lower half. Selecting all columns.
        # DEPENDING ON WHICH CANNY FILTER IS BEST, replace '<image var>' below with that one. This will be the new 'region of interest' (roi)
        roi = auto[120:240, :]

        # Reshape the roi image into one row array
        temp_array = roi.reshape(1, 38400).astype(np.float32)


            # FORWARD
            # Need to pull the directional input at index[frame number]
            image_array = np.vstack((image_array, temp_array))


            # FORWARD_RIGHT
            image_array = np.vstack((image_array, temp_array))


            # FORWARD_LEFT
            image_array = np.vstack((image_array, temp_array))



        # Flip the image_array and label_array to DOUBLE your data.


        # save training images and labels (selects row 1 on down because the row 0 is just zeros)
        train = image_array[1:, :]
        train_labels = label_array[1:, :]

        # save training data as a numpy file
        #''' What exactly does this look like? array of the data & label with 'train' and 'train_labels' as kw/arg? # np.savez(file, *args, **kwargs)'''
        np.savez('training_data_temp/train_{}_NAME.npz'.format(filter_option), train=train, train_labels=train_labels)

        print train.shape
        print train_labels.shape
        print ''
        print 'REMEMBER TO CHANGE THE SAVED npz FILENAME!'
        print ''


if __name__ == '__main__':
    ImageFilterMultiplier()
