__author__ = 'zhengwang  & pseudoyim'

import serial
import pygame
from pygame.locals import *
import car

class RCTest(object):

    def __init__(self):
        pygame.init()
        # self.ser = serial.Serial('/dev/cu.usbmodem1411', 115200, timeout=1)
        self.send_inst = True
        self.duration = 200
        self.steer()


    def steer(self):

        while self.send_inst:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key_input = pygame.key.get_pressed()

                    if key_input[pygame.K_UP]:
                        car.forward(self.duration)

                    elif key_input[pygame.K_DOWN]:
                        car.reverse(self.duration)

                    elif key_input[pygame.K_RIGHT]:
                        car.forward_right(self.duration)
                        car.right(700)

                    elif key_input[pygame.K_LEFT]:
                        car.forward_left(self.duration)
                        car.left(700)

                    # These combination key press events don't work for some reason...
                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                        car.reverse_right(self.duration)
                        car.right(700)

                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                        car.reverse_left(self.duration)
                        car.left(700)


                    # exit
                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'Exit'
                        self.send_inst = False
                        # self.ser.write('00000000\n')
                        # self.ser.close()
                        break

                # elif event.type == pygame.KEYUP:
                #     self.ser.write('00000000\n')

if __name__ == '__main__':
    RCTest()
