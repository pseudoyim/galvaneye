'''
As of 6/22/17
Operation Resurrect-George (just get him to drive, dammit!)
This should be tested on George.
'''

import cv2
import keras.models
import numpy as np
import os
import picamera
import RPi.GPIO as GPIO
import socket
import threading
import time
from imutils.object_detection import non_max_suppression


SIGMA = 0.33
stop_classifier = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
timestr = time.strftime('%Y%m%d_%H%M%S')

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



class CarCamera(object):

    def __init__(self):
        self.camera = picamera.PiCamera()
        self.set_camera_resolution()
        self.frame = None
        self.frame_capture()
        self.start()
        self.


    def set_camera_resolution(self):
        self.camera.resolution = (320, 240)      # pi camera resolution


    def frame_capture(self):
        self.frame = camera.capture('jpeg')      # CONFIRM PARAMETERS
        print 'Captured frame'


    def start(self):
        print 'Starting neural network...'
        NeuralNetwork(receiving=True, CarCameraObject=self)


	def read(self):
		# return the frame most recently read
        print 'Starting frame capture...'
        print ' \"Hold on to your butts!\" '
		return self.frame




class NeuralNetwork(object):

    # global stop_classifier
    global timestr

    def __init__(self, receiving=False, CarCameraObject=None):
        self.receiving = receiving
        self.model = keras.models.load_model('nn_h5/nn.h5')

        # CarCamera class object is now here.
        self.CarCameraObject = CarCameraObject
        self.rcdriver = RCDriver()

        while self.receiving:
            self.fetch()


    def auto_canny(self, blurred):
        # Compute the median of the single channel pixel intensities
        global SIGMA
        v = np.median(blurred)

        # Apply automatic Canny edge detection using the computed median of the image
        lower = int(max(0,   (1.0 - SIGMA) * v))
        upper = int(min(255, (1.0 + SIGMA) * v))
        edged = cv2.Canny(blurred, lower, upper)
        return edged


    def preprocess(self, frame):
        image_array = frame.reshape(1, 38400).astype(np.float32)
        image_array = image_array / 255.    # Normalizing
        return image_array


    def predict(self, image):
        image_array = self.preprocess(image)
        y_hat       = self.model.predict(image_array)
        i_max       = np.argmax(y_hat)
        y_hat_final = np.zeros((1,3))
        np.put(y_hat_final, i_max, 1)
        return y_hat_final[0], y_hat


    def fetch(self):

        frame = 0

        while self.receiving:

            # There's a chance that the script can get to this point before the New thread begins streaming images.
            # To account for this, we create the jpg variable but set to None, and keep checking until it actually has something.
            jpg = None
            while jpg is None:
                jpg = self.CarCameraObject.frame

            gray  = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # Object detection
            obj_detection.detect(stop_classifier, gray, image)

            # Lower half of the grayscale image
            roi = gray[120:240, :]

            # Apply GuassianBlur (reduces noise)
            blurred = cv2.GaussianBlur(roi, (3, 3), 0)

            # Apply Canny filter
            auto = self.auto_canny(blurred)

            # Show streaming images
            cv2.imshow('Original', image)
            cv2.imshow('What the model sees', auto)

            # Neural network model makes prediciton
            # prediction = self.model.predict(auto)
            prediction, probas = self.predict(auto)

            # Save frame and prediction record for debugging research
            prediction_english = None
            prediction_english_proba = None

            proba_left, proba_right, proba_forward = probas[0]

            if np.all(prediction   == [ 0., 0., 1.]):
                prediction_english = 'FORWARD'
                prediction_english_proba = proba_forward

            elif np.all(prediction == [ 1., 0., 0.]):
                prediction_english = 'LEFT'
                prediction_english_proba = proba_left

            elif np.all(prediction == [ 0., 1., 0.]):
                prediction_english = 'RIGHT'
                prediction_english_proba = proba_right

            # cv2.putText(gray, "Model prediction: {}".format(prediction_english), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            cv2.putText(gray, "Prediction (sig={}): {}, {:>05}".format(SIGMA, prediction_english, prediction_english_proba), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            cv2.imwrite('test_frames_temp/frame{:>05}.jpg'.format(frame), gray)
            frame += 1

            # Send prediction to driver to tell it how to steer
            self.rcdriver.steer(prediction)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.receiving = False
                cv2.destroyAllWindows()



class RCDriver(object):

    def __init__(self):
        # self.frame


    def steer(self, prediction):

        # FORWARD
        if np.all(prediction   == [ 0., 0., 1.]):
            print 'forward'
        	GPIO.output(forward, GPIO.LOW)
        	time.sleep(0.3)
            GPIO.output(forward, GPIO.HIGH)

        # FORWARD-LEFT
        elif np.all(prediction == [ 1., 0., 0.]):
            print 'left'
    		GPIO.output(left, GPIO.LOW)
    		GPIO.output(forward, GPIO.LOW)
    		time.sleep(0.4)
    		GPIO.output(left, GPIO.HIGH)
    		GPIO.output(forward, GPIO.HIGH)

        # FORWARD-RIGHT
        elif np.all(prediction == [ 0., 1., 0.]):
            print 'right'
    		GPIO.output(right, GPIO.LOW)
    		GPIO.output(forward, GPIO.LOW)
    		time.sleep(2.0)
    		GPIO.output(right, GPIO.HIGH)
    		GPIO.output(forward, GPIO.HIGH)


    def stop(self):
        print '* * * STOPPING! * * *'
        time.sleep(4.0)

rcdriver = RCDriver()



if __name__ == '__main__':
    try:
        RCDriver()

    except KeyboardInterrupt:
        # Rename the folder that collected all of the test frames. Then make a new folder to collect next round of test frames.
        os.rename(  './test_frames_temp', './test_frames_SAVED/test_frames_{}'.format(timestr))
        os.makedirs('./test_frames_temp')
        print '\nTerminating...\n'

        GPIO.output(left, GPIO.HIGH)
        GPIO.output(right, GPIO.HIGH)
        GPIO.output(forward, GPIO.HIGH)
        GPIO.output(reverse, GPIO.HIGH)
    	GPIO.cleanup()                     # Release resource

        print '\nDone.\n'
