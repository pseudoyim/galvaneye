'''
This module does the following:

'''
import numpy as np
# np.set_printoptions(threshold=np.nan)
import cv2
import glob
import sys
import time
import os, os.path
import shutil
from img_filter_multiply import ImageFilterMultiplier
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm

class ImageAugmenter(object):

    def __init__(self, n=1):

        # DELETE the contents of 'training_images_filtered' and start anew. This is, after all, where the results of this module get dumped.
        shutil.rmtree('./augmented')
        os.makedirs(  './augmented')

        self.blurred = None
        self.n = n

        # Location of original images (or image folders) collected.
        self.loc_originals_img             = './images/imgs_2016*/*.jpg'

        # Location of filtered images after filter is applied to each original image.
        self.loc_filtered_img_storage_each = 'training_images_filtered/frame{:>05}.jpg'

        # Location of filtered images, referred to when 'multiply()' method commences, so it knows which images to multiply.
        self.loc_filtered_img_storage      = 'training_images_filtered/*.jpg'

        # Location of original label_array (to be multiplied).
        self.loc_originals_label_array     = './images/imgs_2016*/label_array_ORIGINALS.npz'

        # Location where final npz file will be saved (after filter application and multiplication are finished).
        # self.loc_final_save                = 'training_data_temp/aug_sigma{}_{}.npz'.format(str(self.sigma)[-2:], time.strftime("%Y%m%d_%H%M%S"))

        if self.n == None:
            print 'You want to augment, but you didn\'t give me an "n" parameter. Try again, fool.'
            sys.exit()
        self.augment()
        ImageFilterMultiplier(sigma=0.33, subsequent=True, augment=True, n=self.n)


    def augment(self):
        print ''
        print '*** AUGMENT ***'
        print 'Generate new augmented images: Initiated'

        # Folders with original images, read them in one-by-one. They'll be in order according to the filepaths, which are ordered by timestamp.
        originals = glob.glob(self.loc_originals_img)

        print 'Generating {} augmentations for each of {} original images...'.format(self.n, len(originals))

        datagen = ImageDataGenerator(rotation_range=1,
                                     width_shift_range=0.02,
                                     height_shift_range=0.02,
                                     shear_range=0,
                                     zoom_range=0.1,
                                     horizontal_flip=False,
                                     fill_mode='nearest')

        successes = 0
        for orig in tqdm(originals):
            # Add a copy of original to './augmented' folder before adding its augmented copies.
            shutil.copy(orig, './augmented')
            os.rename('./augmented/{}'.format(orig[-14:]), './augmented/aug_{}_{:>05}_orig.jpg'.format(orig[-30:-15],successes))
            successes += 1

            # For each original .jpg in folder
            img = load_img(orig)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for i in xrange(self.n):

                i = 0
                for batch in datagen.flow(x,
                                          batch_size=1,
                                          save_to_dir='./augmented',
                                          save_prefix='aug_{}_{:>05}'.format(orig[-30:-15],successes),
                                          save_format='jpg'):
                    i += 1
                    successes += 1
                    if i == 1:
                        i = 0
                        break

        print '...complete!'

        # HACKY: Check if number of images in save_to_dir == number of successes.
        num_new_images = len([name for name in os.listdir('./augmented')])
        if successes != num_new_images:
            print 'Number of successes:', successes
            print 'Number of augmented images:', num_new_images
            print successes, '!=', num_new_images
            print 'FAIL! Numbers don\'t match.'
            sys.exit()

        print 'Generate new augmented images: Completed'
        print ''
        print 'SUMMARY:'
        print 'Number of original images:', len(originals)
        print 'Number of original images + augmented images:', num_new_images


if __name__ == '__main__':

    ImageAugmenter(n=2)
    ImageFilterMultiplier(sigma=0.33, subsequent=True, augment=True, n=)
