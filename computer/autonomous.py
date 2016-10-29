import car
import cv2
import numpy as np
import serial
import socket
import SocketServer
import threading

# KERAS stuff
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.models

SIGMA = 0.25


class NeuralNetwork(object):

    def __init__(self):
        self.model = keras.models.load_model('nn_h5/nn.h5')

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



class RCDriver(object):

    def steer(self, prediction):

        # FORWARD
        if np.all(prediction   == [ 0., 0., 1.]):
            car.forward(100)
            car.pause(500)
            print("Forward")

        # FORWARD-LEFT
        elif np.all(prediction == [ 1., 0., 0.]):
            car.left(300)
            car.forward_left(200)
            car.left(700)
            car.pause(500)
            print("Left")

        # FORWARD-RIGHT
        elif np.all(prediction == [ 0., 1., 0.]):
            car.right(300)
            car.forward_right(200)
            car.right(700)
            car.pause(500)
            print("Right")



class VideoStreamHandler(object):

    model  = NeuralNetwork()
    driver = RCDriver()

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('192.168.1.66', 8000)) # The IP address of your computer (Paul's MacBook Air). This script should run before the one on the Pi.
        print 'Listening...'
        self.server_socket.listen(0)

        # Accept a single connection ('rb' is 'read binary')
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # Establish a condition that RaspPi should be sending images.
        self.send_inst = True

        # Start handling video feed, predict, and drive
        self.handle()


    def auto_canny(self, blurred):
        # Compute the median of the single channel pixel intensities
        global SIGMA
        v = np.median(blurred)

        # Apply automatic Canny edge detection using the computed median of the image
        lower = int(max(0,   (1.0 - SIGMA) * v))
        upper = int(min(255, (1.0 + SIGMA) * v))
        edged = cv2.Canny(blurred, lower, upper)
        return edged


    def handle(self):

        model  = NeuralNetwork()
        driver = RCDriver()

        # Stream video frames one by one.
        try:
            stream_bytes = ' '
            while self.send_inst:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')           # ? What is this string and where did it come from?
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]

                    gray  = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    # image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    # Lower half of the grayscale image
                    roi = gray[120:240, :]

                    # Apply GuassianBlur (reduces noise)
                    blurred = cv2.GaussianBlur(roi, (3, 3), 0)

                    # Apply Canny filter
                    auto = self.auto_canny(blurred)

                    # Show streaming images
                    cv2.imshow('What the model sees', auto)
                    # cv2.imshow('Original', image)

                    # Neural network model makes prediciton
                    prediction = model.predict(auto)

                    # Send prediction to driver to tell it how to steer
                    driver.steer(prediction)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.send_inst = False
                        break

            cv2.destroyAllWindows()

        finally:
            self.connection.close()
            self.server_socket.close
            print 'Connection closed'
            print 'Socket closed'


if __name__ == '__main__':

    print '\n \"Hold on to your butts.\" \n'
    VideoStreamHandler()
