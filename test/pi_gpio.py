import RPi.GPIO as GPIO
import time

voltage = 1
ground  = 9

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


try:
	while True:

        print 'forward'
		GPIO.output(forward, GPIO.LOW)
		time.sleep(2.0)
        GPIO.output(forward, GPIO.HIGH)

        print 'reverse'
		GPIO.output(reverse, GPIO.LOW)
		time.sleep(2.0)
        GPIO.output(reverse, GPIO.HIGH)

        print 'left'
		GPIO.output(left, GPIO.LOW)
		time.sleep(2.0)
        GPIO.output(left, GPIO.HIGH)

        print 'right'
		GPIO.output(right, GPIO.LOW)
		time.sleep(2.0)
        GPIO.output(right, GPIO.HIGH)

        print 'left-forward'
		GPIO.output(left, GPIO.LOW)
		GPIO.output(forward, GPIO.LOW)
		time.sleep(2.0)
		GPIO.output(left, GPIO.HIGH)
		GPIO.output(forward, GPIO.HIGH)

        print 'right-forward'
		GPIO.output(right, GPIO.LOW)
		GPIO.output(forward, GPIO.LOW)
		time.sleep(2.0)
		GPIO.output(right, GPIO.HIGH)
		GPIO.output(forward, GPIO.HIGH)

except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the flowing code will be  executed.
    GPIO.output(left, GPIO.HIGH)
    GPIO.output(right, GPIO.HIGH)
    GPIO.output(forward, GPIO.HIGH)
    GPIO.output(reverse, GPIO.HIGH)
	GPIO.cleanup()                     # Release resource
