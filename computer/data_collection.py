__author__ = 'zhengwang & pseudoyim'

import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import car
import argparse
import glob

class CollectTrainingData(object):

    '''
    Hamuchiwa used this class to stream video from Pi on the car to collect training images and user input, resulting in paired image data and label arrays saved into a npz file.
    *** THIS SCRIPT SHOULD RUN BEFORE THE 'stream_client.py' ON THE PI!!! ***
    '''

    def __init__(self):

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('192.168.1.66', 8000)) # The IP address of your computer (Paul's MacBook Air). This script should run before the one on the Pi.
        print 'Listening...'
        self.server_socket.listen(0)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # # connect to a serial port
        # (Hamuchiwa's old code) self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)
        # self.ser = serial.Serial('/dev/cu.usbmodem1411', 115200, timeout=1)     # ? How exactly did this port get created?  Obtained port path from: python -m serial.tools.list_ports
        self.send_inst = True

        # create labels (aka the Y values; these will be the directional output to the arduino remote control)
        # Creates a 4x4 matrix, with 1's along the diagonal, upper left to bottom right:
        # array([[ 1.,  0.,  0.,  0.],
        #        [ 0.,  1.,  0.,  0.],
        #        [ 0.,  0.,  1.,  0.],
        #        [ 0.,  0.,  0.,  1.]])
        self.k = np.zeros((3, 3), 'float')
        for i in range(3):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 3), 'float')

        pygame.init()
        self.collect_image()
        self.auto_canny()


    def auto_canny(self,img):
    	# compute the median of the single channel pixel intensities
        sigma=0.33
        v = np.median(img)

    	# apply automatic Canny edge detection using the computed median
    	lower = int(max(0, (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	edged = cv2.Canny(img, lower, upper)

    	# return the edged image
    	return edged


    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        clicks_forward       = 0
        clicks_forward_left  = 0
        clicks_forward_right = 0
        clicks_reverse       = 0

        # collect images for training
        print 'Start collecting images...'
        e1 = cv2.getTickCount()

        # image_array and label_array each start off as one row of zeros. As images are added, they are vstacked from below.
        image_array = np.zeros((1, 38400))      # Image resolution is 320x240. But we are lopping off the top half, so actually 320x120. 320 * 120 = 38400.
        label_array = np.zeros((1, 3), 'float')

        # stream video frames one by one
        try:
            stream_bytes = ' '
            frame = 1
            while self.send_inst:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')           # ? What is this string and where did it come from?
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]

                    # image is a np matrix.
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    # overlay = image.copy()  IF putText works below, then delete this line. Otherwise you'll need to add a transparent overlay (cv2.addWeighted(overlay...)
                    blurred = cv2.GaussianBlur(image, (3, 3), 0)


                    # images with Canny filter applied:
                    wide = cv2.Canny(blurred, 10, 200)
                    tight = cv2.Canny(blurred, 225, 250)
                    auto = self.auto_canny(blurred)

                    # select lower half of the image. 0:120 would be the upper half of rows. 120:240 is the lower half. Selecting all columns.
                    # DEPENDING ON WHICH CANNY FILTER IS BEST, replace '<image var>' below with that one. This will be the new 'region of interest' (roi)
                    roi = auto[120:240, :]

                    # overlay click counts: cv2.putText(clicks_*)
                    cv2.putText(image, "FW: {}, LT: {}, RT: {}, REV: {}".format(clicks_forward, clicks_forward_left, clicks_forward_right, clicks_reverse), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)

                    # save streamed images
                    # DEPENDING ON WHICH CANNY FILTER IS BEST, replace '<image var>' below with that one. These will be the new saved streamed images.
                    cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), auto)

                    # Display feeds on host (laptop)
                    cv2.imshow('image', image)
                    cv2.imshow('roi_image', roi)
                    # cv2.imshow('original image & edge', np.hstack([image, auto]))

                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)

                    frame += 1
                    total_frame += 1

                    # get input from human driver
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:                                   #''' 'KEYDOWN' : Does this just indicate when a key is pressed DOWN? '''
                            key_input = pygame.key.get_pressed()

                            # FORWARD
                            if key_input[pygame.K_UP]:
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))   # self.k[2] = [ 0.,  0.,  1.]
                                saved_frame += 1
                                clicks_forward += 1
                                car.forward(200)

                            # FORWARD_RIGHT
                            elif key_input[pygame.K_RIGHT]:
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))   # self.k[1] = [ 0.,  1.,  0.]
                                saved_frame += 1
                                clicks_forward_right += 1
                                car.right(300)
                                car.forward_right(300)
                                car.right(700)

                            # FORWARD_LEFT
                            elif key_input[pygame.K_LEFT]:
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))   # self.k[0] = [ 1.,  0.,  0.]
                                saved_frame += 1
                                clicks_forward_left += 1
                                car.left(300)
                                car.forward_left(300)
                                car.left(700)

                            # # REVERSE; NOT USING
                            # elif key_input[pygame.K_DOWN]:
                            #     image_array = np.vstack((image_array, temp_array))
                            #     label_array = np.vstack((label_array, self.k[3]))   # self.k[3] = [ 0.,  0.,  0.,  1.]
                            #     saved_frame += 1
                            #     clicks_reverse += 1
                            #     car.reverse(200)


                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                self.send_inst = False
                                # self.ser.write(chr(0))                              # chr(0) == 00000000
                                break

                        elif event.type == pygame.KEYUP:                            #'''Key is lifted UP?'''
                            # self.ser.write(chr(0))                                  # chr(0) == 00000000
                            break

            # save training images and labels (selects row 1 on down because the row 0 is just zeros)
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            #''' What exactly does this look like? array of the data & label with 'train' and 'train_labels' as kw/arg? # np.savez(file, *args, **kwargs)'''
            np.savez('training_data_temp/testNAME.npz', train=train, train_labels=train_labels)
            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print 'Streaming duration:', time0

            print(train.shape)
            print(train_labels.shape)
            print 'Total frame:', total_frame
            print 'Saved frame:', saved_frame
            print 'Dropped frame', total_frame - saved_frame

        finally:
            self.connection.close()
            self.server_socket.close
            print 'Connection closed'
            print 'Socket closed'

if __name__ == '__main__':
    CollectTrainingData()
