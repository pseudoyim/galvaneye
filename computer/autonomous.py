import threading
import SocketServer
import serial
import cv2
import numpy as np
import math
import car

# KERAS stuff
from keras.models import Sequential
from keras.layers import Dense, Activation

SIGMA = 0.33


class NeuralNetwork(object):

    def __init__(self):
        self.model = load_model('nn_h5/nn.h5')

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
        image = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        roi = image[120:240, :]
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        auto = self.auto_canny(blurred)
        image_array = auto.reshape(1, 38400).astype(np.float32)
        image_array = image_array / 255.
        return image_array

    def predict(self, frame):
        image_array = self.preprocess(frame)
        y_hat       = self.model.predict(image_array)
        i_max       = np.argmax(y_hat)
        y_hat_final = np.zeros((1,3))
        np.put(y_hat_final, i_max, 1)
        return y_hat_final



class RCControl(object):
    # Needs to be adapted for 'car' module

    def __init__(self):
        # NEED TO UPDATE THIS
        self.serial_port = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)

    def steer(self, prediction):
        if prediction == 2:
            self.serial_port.write(chr(1))
            print("Forward")
        elif prediction == 0:
            self.serial_port.write(chr(7))
            print("Left")
        elif prediction == 1:
            self.serial_port.write(chr(6))
            print("Right")
        else:
            self.stop()

    def stop(self):
        self.serial_port.write(chr(0))



class VideoStreamHandler(SocketServer.StreamRequestHandler):

    model  = NeuralNetwork()
    rc_car = RCControl()

    def handle(self):

        global sensor_data
        stream_bytes = ' '
        stop_flag = False
        stop_sign_active = True

        # Stream video frames
        try:
            while True:
                stream_bytes += self.rfile.read(1024)               # 'self.rfile' is a file-like object created by the handler
                first = stream_bytes.find('\xff\xd8')               #'''Again, what are these strings?'''
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)    #'''might be deprecated'''
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)

                    # Lower half of the streamed image
                    half_gray = gray[120:240, :]

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    image_array = half_gray.reshape(1, 38400).astype(np.float32)

                    # NEURAL NETWORK MAKES PREDICTION
                    prediction = self.model.predict(image_array)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        # self.rc_car.stop()
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on video stream thread"



class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread('192.168.1.66', 8000))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
