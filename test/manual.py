__author__ = 'pseudoyim'

import pygame
from pygame.locals import *
import car

class ManualDrive(object):

    def __init__(self):

        self.send_inst = True
        self.duration = 200
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
                        car.forward(self.duration)

                    # FORWARD_RIGHT
                    elif key_input[pygame.K_RIGHT]:
                        car.right(200)
                        car.forward_right(self.duration)
                        car.right(700)

                    # FORWARD_LEFT
                    elif key_input[pygame.K_LEFT]:
                        car.left(200)
                        car.forward_left(self.duration)
                        car.left(700)

                    # REVERSE
                    elif key_input[pygame.K_DOWN]:
                        car.reverse(self.duration)


                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        self.send_inst = False
                        break

                elif event.type == pygame.KEYUP:                            #'''Key is lifted UP?'''
                    break


if __name__ == '__main__':
    ManualDrive()
