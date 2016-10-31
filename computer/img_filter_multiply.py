'''
This module does the following:

'''
import cv2
import glob
import numpy as np
# np.set_printoptions(threshold=np.nan)
import os
import shutil
import sys
import time
# from img_augment import ImageAugmenter
from tqdm import tqdm

class ImageFilterMultiplier(object):

    # 'subsequent' refers to whether this run is being performed (subsequently) on previously saved original images, but at a different sigma value.
    def __init__(self, sigma=0.33, subsequent=False, augment=False, n=None, timestr=None):

        if not subsequent:
            confirmation1 = raw_input('Confirmed the directory \'training_images\' contains the ORIGINAL images you want to filter & multiply? [y/n] ')
            if confirmation1 != 'y':
                sys.exit()

        # DELETE the contents of 'training_images_filtered'
        shutil.rmtree('./training_images_filtered')
        os.makedirs(  './training_images_filtered')

        self.augment = augment
        self.blurred = None
        self.n = n
        self.sigma = sigma
        self.subsequent = False
        self.timestr = timestr

        if subsequent:
            self.subsequent = True

        # Location of images to filter and multiply (after making a copy and flipping it L to R).
        if subsequent and not augment:
            self.loc_originals_img         = './images/imgs_2016*/*.jpg'
        elif subsequent and augment:
            self.loc_originals_img         = './augmented/*.jpg'
        else:
            self.loc_originals_img         = './training_images/*.jpg'

        # Location to store filtered images after filter is applied to each original image.
        self.loc_filtered_img_storage_each = './training_images_filtered/frame{:>05}.jpg'

        # Location of filtered images, referred to when 'multiply()' method commences, so it knows which images to multiply.
        self.loc_filtered_img_storage      = './training_images_filtered/*.jpg'

        # Location of original label_array (to be multiplied).
        if subsequent:
            self.loc_originals_label_array = './images/imgs_2016*/label_array_ORIGINALS.npz'    # This is correct path for both subsequent=True AND augment=True
        else:
            self.loc_originals_label_array = './training_images/label_array_ORIGINALS.npz'
        # self.loc_originals_label_array     = './training_images/label_array_SUBSET.npz'

        # Location where final npz file will be saved (after filter application and multiplication are finished).
        if augment:
            # self.loc_final_save            = './training_data_temp/aug_sigma{}_{}.npz'.format(str(self.sigma)[-2:], time.strftime("%Y%m%d_%H%M%S"))
            self.loc_final_save            = './training_data_temp/aug_sigma{}_{}.npz'.format(str(self.sigma)[-2:], self.timestr)

        else:
            # self.loc_final_save            = './training_data_temp/sigma{}_{}.npz'.format(str(self.sigma)[-2:], time.strftime("%Y%m%d_%H%M%S"))
            self.loc_final_save            = './training_data_temp/sigma{}_{}.npz'.format(str(self.sigma)[-2:], self.timestr)


        self.apply_filter()
        self.multiply()



    def auto_canny(self):
    	# Compute the median of the single channel pixel intensities
        sigma = self.sigma
        v = np.median(self.blurred)

    	# Apply automatic Canny edge detection using the computed median
    	lower = int(max(0,   (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	edged = cv2.Canny(self.blurred, lower, upper)

    	# Return the edged image
    	return edged



    def apply_filter(self):

        print ''
        print '*** FILTER (at sigma={}) ***'.format(self.sigma)
        print 'Apply filter to original images: Initiated'

        frame = 0

        # original images, read them in one-by-one
        originals = glob.glob(self.loc_originals_img)

        for each in tqdm(originals):

            # image is a np matrix.
            image = cv2.imread(each, cv2.IMREAD_GRAYSCALE)

            # select lower half of the image. 0:120 would be the upper half of rows. 120:240 is the lower half. Selecting all columns.
            roi = image[120:240, :]

            self.blurred = cv2.GaussianBlur(roi, (3, 3), 0)

            # Different Canny filter parameters
            auto = self.auto_canny()

            # Save filtered images
            cv2.imwrite(self.loc_filtered_img_storage_each.format(frame), auto)
            frame +=1

        print 'Apply filter to original images: Completed'
        print ''



    def multiply(self):

        print '*** MULTIPLY ***'
        print 'Multiply (double) filtered images and labels: Initiated'

        # image_array and label_array each start off as one row of zeros. As images are added, they are vstacked from below.
        image_array = []      # Image resolution is 320x240 (width & height). But we are lopping off the top half, so actually 320x120. 320 * 120 = 38400.
        label_array = []

        # *** HANDLES UPDATE OF 'image_array' ***
        # Filtered images, read them in one-by-one
        originals = glob.glob(self.loc_filtered_img_storage)

        print 'Copying/flipping images, converting to arrays, and compiling into final list...'
        for each in tqdm(originals):                        # single_npz == one npz file in the 'training_data' folder.

            image = cv2.imread(each, cv2.IMREAD_GRAYSCALE)

            # Decode/reshape ORIGINAL image into np.array
            temp_array = image.reshape(1, 38400).astype(np.float32)
            image_array.append(temp_array)

            # Flip temp_array (i.e. left will be right, right will be left, forward will be forward)
            # Decode/reshape FLIPPED image into np.array
            image_flipped = np.fliplr(image)
            temp_array_flipped = image_flipped.reshape(1, 38400).astype(np.float32)
            image_array.append(temp_array_flipped)

        image_array = np.array(image_array)

        #HACKY: To solve the extra dimension problem...
        num_new_rows = image_array.shape[0]
        image_array = image_array.reshape(num_new_rows, 38400).astype(np.float32)


        print '...complete!'

        # *** HANDLES UPDATE OF 'label_array' ***
        # Flip the LEFT & RIGHT directional inputs, but leave FORWARD alone. (for directional target/label data)
        # array([[ 1.,  0.,  0.],   <-- FORWARD-LEFT
        #        [ 0.,  1.,  0.],   <-- FORWARD-RIGHT
        #        [ 0.,  0.,  1.]])  <-- FORWARD
        print 'Updating label_arrays...'
        original_labels_shape = None
        labels_doubled = None

        # Finds the npz file for the label_array when the ORIGINAL images were collected
        training_data_labels = glob.glob(self.loc_originals_label_array)

        # This 'final' empty list will eventually be turned into a nparray
        final = []

        print '...{} existing label_array files to go through...'.format(len(training_data_labels))
        for single_npz in tqdm(training_data_labels):

            with np.load(single_npz) as data:

                # ORIGINAL LABELS
                labels = data.f.train_labels
                original_labels_shape = labels.shape

                array_list = labels.tolist()

                # For each original label...
                for row in array_list:

                    if self.augment:

                        for i in xrange(self.n + 1):
                            final.append(row)

                            # Forward-Left becomes Forward-Right, append
                            if row[0] == 1:
                                final.append([0.0, 1.0, 0.0])

                            # Forward-Right becomes Forward-Left, append
                            elif row[1] == 1:
                                final.append([1.0, 0.0, 0.0])

                            # Forward remains as Forward, append
                            elif row[2] == 1:
                                final.append([0.0, 0.0, 1.0])
                                continue

                    else:
                        # First append original row to final, then...
                        final.append(row)

                        # Forward-Left becomes Forward-Right, append
                        if row[0] == 1:
                            row = [0.0, 1.0, 0.0]
                            final.append(row)

                        # Forward-Right becomes Forward-Left, append
                        elif row[1] == 1:
                            row = [1.0, 0.0, 0.0]
                            final.append(row)

                        # Forward remains as Forward, append
                        elif row[2] == 1:
                            final.append(row)
                            continue

        label_array = np.array(final)

        print '...complete!'
        print ''

        # save training images and labels (selects row 1 on down because the row 0 is just zeros)
        train = image_array
        train_labels = label_array

        # save training data as a numpy array zip file
        #''' What exactly does this look like? array of the data & label with 'train' and 'train_labels' as kw/arg? # np.savez(file, *args, **kwargs)'''
        print 'Saving final .npz file...'
        np.savez(self.loc_final_save, train=train, train_labels=train_labels)

        # Save new images in folder
        # if not subsequent (i.e. if this is a brand new set of training images), then save to its own new folder.
        if not self.subsequent:
            os.rename(  './training_images', './imgs_{}'.format(self.timestr))
            os.makedirs('./training_images')

        print 'Multiply (double) filtered images and labels: Completed'
        print ''

        print 'Initial number of original images: ', len(originals)
        print 'Initial label_array dims (the last one opened): ', original_labels_shape
        print 'Final image_array dims: ', train.shape
        print 'Final label_array dims: ', train_labels.shape
        print ''
        print 'Filter and Multiply successfully completed.'
        print ''
        print 'REMEMBER: Upload final training data npz files to Amazon S3.'
        print ''



if __name__ == '__main__':

    # ImageFilterMultiplier(sigma=0.33, subsequent=False)   # USE THIS WHEN COLLECTING NEW TRAINING DATA; this is the only sigma for whihch subsequent=False
    # ImageFilterMultiplier(sigma=0.20, subsequent=True)
    # ImageFilterMultiplier(sigma=0.46, subsequent=True)
    pass
