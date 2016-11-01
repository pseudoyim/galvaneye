import car
import cv2
import numpy as np
import os
import serial
import socket
import SocketServer
import threading
import time

# KERAS stuff
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.models

SIGMA = 0.25
stop_classifier = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
timestr = time.strftime('%Y%m%d_%H%M%S')


class RCDriver(object):

    def steer(self, prediction):

        # FORWARD
        if np.all(prediction   == [ 0., 0., 1.]):
            car.forward(100)
            car.pause(500)
            print 'Forward'

        # FORWARD-LEFT
        elif np.all(prediction == [ 1., 0., 0.]):
            car.left(300)
            car.forward_left(200)
            car.left(700)
            car.pause(500)
            print 'Left'

        # FORWARD-RIGHT
        elif np.all(prediction == [ 0., 1., 0.]):
            car.right(300)
            car.forward_right(200)
            car.right(700)
            car.pause(500)
            print 'Right'

    def stop(self):
        print '*** STOPPING! ***'
        car.pause(8000)
        car.forward(100)



class ObjectDetection(object):

    global rcdriver
    global stop_classifier

    def detect(self, cascade_classifier, gray_image, image):


        stop_sign_detected = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25),
            maxSize=(55, 55)
        )

        rect_width = None

        # Draw a rectangle around stop sign
        for (x_pos, y_pos, width, height) in stop_sign_detected:
            rect_width = width
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (0, 0, 255), 2)
            cv2.putText(image, 'STOP SIGN', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # Execute the full stop
        if np.any(stop_sign_detected):
            rcdriver.stop()



class NeuralNetwork(object):

    global stop_classifier
    global timestr
    obj_detection = ObjectDetection()
    rcdriver = RCDriver()


    def __init__(self, receiving=False):
        self.receiving = receiving
        self.model = keras.models.load_model('nn_h5/nn.h5')

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
        image_array = image_array / 255.
        return image_array


    def predict(self, image):
        image_array = self.preprocess(image)
        y_hat       = self.model.predict(image_array)
        i_max       = np.argmax(y_hat)
        y_hat_final = np.zeros((1,3))
        np.put(y_hat_final, i_max, 1)
        return y_hat_final[0]


    def fetch(self):

        while self.receiving:

            frame = 0

            jpg = video_stream.frame

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
            prediction = self.model.predict(auto)

            # Save frame and prediction record for debugging research
            prediction_english = None
            if np.all(prediction   == [ 0., 0., 1.]):
                prediction_english = 'FORWARD'
            elif np.all(prediction == [ 1., 0., 0.]):
                prediction_english = 'LEFT'
            elif np.all(prediction == [ 0., 1., 0.]):
                prediction_english = 'RIGHT'

            cv2.putText(gray, "Model prediction: {}".format(prediction_english), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            cv2.imwrite('test_frames_temp/frame{:>05}.jpg'.format(frame), gray)
            frame += 1

            # Send prediction to driver to tell it how to steer
            rcdriver.steer(prediction)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                cv2.destroyAllWindows()

                # Rename the folder that collected all of the test frames. Then make a new folder to collect next round of test frames.
                os.rename(  './test_frames_temp', './test_frames_SAVED/test_frames_{}'.format(timestr))
                os.makedirs('./test_frames_temp')




class PiVideoStream(object):


    neural_network = NeuralNetwork()

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('192.168.1.66', 8000)) # The IP address of your computer (Paul's MacBook Air). This script should run before the one on the Pi.
        print 'Listening...'
        self.server_socket.listen(0)

        # Accept a single connection ('rb' is 'read binary')
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.stream_bytes = ' '

        self.start()



    def start(self):
    	# start the thread to read frames from the video stream
        print 'Starting PiVideoStream thread...'
        threading.Thread(target=self.update, args=()).start()

        print '...thread running'
        # neural_network

        return self


    def update(self):
        while True:
            self.stream_bytes += self.connection.read(1024)
            first = self.stream_bytes.find('\xff\xd8')
            last = self.stream_bytes.find('\xff\xd9')
            if first != -1 and last != -1:
                self.frame = self.stream_bytes[first:last + 2]
                self.stream_bytes = self.stream_bytes[last + 2:]

		# if the thread indicator variable is set, stop the thread
		# and resource camera resources
		if self.stopped:
			self.connection.close()
			return


	def read(self):
		# return the frame most recently read
		return self.frame


	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True



if __name__ == '__main__':
    PiVideoStream()
    print ' \"Hold on to your butts!\" '
    # NeuralNetwork(receiving=True)
