import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import car
import argparse
import glob

class CollectNegImgData(object):

    '''
    Use this script to collect negative images of the road (for Haar Cascade training).
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


    def collect_image(self):

        total_frame = 0

        # collect images for training
        print 'Start collecting NEGATIVE images...'
        e1 = cv2.getTickCount()

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
                    image_framecount = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                    # overlay click counts: cv2.putText(clicks_*)
                    cv2.putText(image_framecount, "Total frames: {}".format(total_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)

                    # save streamed images
                    cv2.imwrite('negative_images/frame{:>05}.jpg'.format(frame), image)

                    # Display feeds on host (laptop)
                    cv2.imshow('image', image)
                    cv2.imshow('image_framecount', image_framecount)

                    frame += 1
                    total_frame += 1

                    # get input from human driver
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:                                   #''' 'KEYDOWN' : Does this just indicate when a key is pressed DOWN? '''
                            key_input = pygame.key.get_pressed()

                            # FORWARD
                            if key_input[pygame.K_UP]:
                                car.forward(300)

                            # FORWARD_RIGHT
                            elif key_input[pygame.K_RIGHT]:
                                car.right(300)
                                car.forward_right(400)
                                car.right(700)

                            # FORWARD_LEFT
                            elif key_input[pygame.K_LEFT]:
                                car.left(300)
                                car.forward_left(400)
                                car.left(700)

                            # REVERSE
                            elif key_input[pygame.K_DOWN]:
                                car.reverse(300)

                            # Quit
                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                self.send_inst = False
                                break

                        elif event.type == pygame.KEYUP:                            #'''Key is lifted UP?'''
                            break


            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print 'Streaming duration:', time0

            print 'Total NEGATIVE frames collected:', total_frame


        finally:
            self.connection.close()
            self.server_socket.close
            print 'Connection closed'
            print 'Socket closed'

if __name__ == '__main__':
    CollectNegImgData()
