__author__ = 'pseudoyim'

import pygame
from pygame.locals import *
import car

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
                        car.forward(200)

                    # FORWARD_RIGHT
                    elif key_input[pygame.K_RIGHT]:
                        car.right(300)
                        car.forward_right(300)
                        car.right(500)

                    # FORWARD_LEFT
                    elif key_input[pygame.K_LEFT]:
                        car.left(300)
                        car.forward_left(300)
                        car.left(500)

                    # REVERSE
                    elif key_input[pygame.K_DOWN]:
                        car.reverse(200)


                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        self.send_inst = False
                        break

                elif event.type == pygame.KEYUP:                            #'''Key is lifted UP?'''
                    break


if __name__ == '__main__':
    ManualDrive()
