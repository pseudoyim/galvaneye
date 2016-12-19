__author__ = 'pseudoyim'

import pygame
import RPi.GPIO as GPIO
from pygame.locals import *

left    = 11
right   = 12
forward = 13
reverse = 15

GPIO.setmode(GPIO.BOARD)       # Numbers pins by physical location

GPIO.setup(left, GPIO.OUT)
GPIO.output(left, GPIO.HIGH)

GPIO.setup(right, GPIO.OUT)
GPIO.output(right, GPIO.HIGH)

GPIO.setup(forward, GPIO.OUT)
GPIO.output(forward, GPIO.HIGH)

GPIO.setup(reverse, GPIO.OUT)
GPIO.output(reverse, GPIO.HIGH)



class ManualDrive(object):

    def __init__(self):

        self.send_inst = True
        pygame.init()
        self.drive()

    def drive(self):

        while self.send_inst:

            # get input from human driver
            for event in pygame.event.get():
                if event.type == KEYDOWN:                                   #''' 'KEYDOWN' : Does this just indicate when a key is pressed DOWN? '''
                    key_input = pygame.key.get_pressed()

                    # FORWARD
                    if key_input[pygame.K_UP]:
                        GPIO.output(forward, GPIO.LOW)

                    # FORWARD_RIGHT
                    elif key_input[pygame.K_RIGHT]:
                		GPIO.output(right, GPIO.LOW)
                		GPIO.output(forward, GPIO.LOW)

                    # FORWARD_LEFT
                    elif key_input[pygame.K_LEFT]:
                    	GPIO.output(left, GPIO.LOW)
                    	GPIO.output(forward, GPIO.LOW)

                    # REVERSE
                    elif key_input[pygame.K_DOWN]:
                        GPIO.output(reverse, GPIO.LOW)


                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        self.send_inst = False
                        break

                elif event.type == pygame.KEYUP:                            #'''Key is lifted UP?'''
                    break


if __name__ == '__main__':
    ManualDrive()
