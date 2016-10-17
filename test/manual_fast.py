import curses
import time
import sys
import car

forward, reverse, left, right = False, False, False, False

error = None

screen = curses.initscr()
screen.keypad(True)

try:

    while True:
        char = screen.getch()
        status = 'nothing'
        if char == 113: raise Exception('quit')  # q
        elif char == curses.KEY_RIGHT:
            if left: left = False
            else: right = not right
            status = 'right'
        elif char == curses.KEY_LEFT:
            if right: right = False
            else: left = not left
            status = 'left'
        elif char == curses.KEY_UP:
            if reverse: reverse = False
            else: forward = not forward
            status = 'up'
        elif char == curses.KEY_DOWN:
            if forward: forward = False
            else: reverse = not reverse
            status = 'down'
        elif char == 32:
            forward, reverse, left, right = False, False, False, False
            status = 'STOP!'
        else:
            status = str(char)
        screen.clear()
        screen.addstr(0, 0, status)
        screen.refresh()
        car.set(forward, reverse, left, right)

except BaseException as e:
    error = e

curses.endwin()

print error
