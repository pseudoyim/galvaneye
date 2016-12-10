import car
import serial
import serial.tools.list_ports
import time

port_to_use = None
for port in serial.tools.list_ports.comports():
    if 'uno' in port.description.lower():
        port_to_use = port
        break

ser = serial.Serial(port.device, 9600)

print ser.readline()

def forward(ms):
    ser.write('1000\n')
    # print 'forward'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def reverse(ms):
    ser.write('0100\n')
    # print 'reverse'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def left(ms):
    ser.write('0010\n')
    # print 'left'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def right(ms):
    ser.write('0001\n')
    # print 'right'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def forward_left(ms):
    ser.write('1010\n')
    # print 'forward_left'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def forward_right(ms):
    ser.write('1001\n')
    # print 'forward_right'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def reverse_left(ms):
    ser.write('0110\n')
    # print 'reverse_left'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def reverse_right(ms):
    ser.write('0101\n')
    # print 'reverse_right'
    time.sleep(ms / 1000.)
    ser.write('0000\n')

def stop():
    ser.write('0000\n')

def set(forward, reverse, left, right):
    forward = 1 if forward else 0
    reverse = 1 if reverse else 0
    left = 1 if left else 0
    right = 1 if right else 0
    ser.write("{}{}{}{}\n".format(forward, reverse, left, right))

def pause(ms):
    # print 'pausing...'
    time.sleep(ms / 1000.)
