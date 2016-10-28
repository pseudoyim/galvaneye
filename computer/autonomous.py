import car
import cv2
import numpy as np
import serial
import SocketServer
import threading

# KERAS stuff
from keras.layers import Dense, Activation
from keras.models import Sequential

SIGMA = 0.33


class NeuralNetwork(object):

    def __init__(self):
        self.model = load_model('nn_h5/nn.h5')

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
        if prediction   == [ 0.  0.  1.] :
            car.forward(200)
            print("Forward")

        # FORWARD-LEFT
        elif prediction == [ 1.  0.  0.] :
            car.left(300)
            car.forward_left(300)
            car.left(700)
            print("Left")

        # FORWARD-RIGHT
        elif prediction == [ 0.  1.  0.] :
            car.right(300)
            car.forward_right(300)
            car.right(700)
            print("Right")



class VideoStreamHandler(SocketServer.StreamRequestHandler):

    model  = NeuralNetwork()
    driver = RCDriver()

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

        # Stream video frames one by one.
        try:
            stream_bytes = ' '
            while True:
                stream_bytes += self.rfile.read(1024)               # 'self.rfile' is a file-like object created by the handler
                first = stream_bytes.find('\xff\xd8')               # What are these strings?
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]

                    gray  = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    # Lower half of the grayscale image
                    roi = gray[120:240, :]

                    # Apply GuassianBlur (reduces noise)
                    blurred = cv2.GaussianBlur(roi, (3, 3), 0)

                    # Apply Canny filter
                    auto = self.auto_canny(blurred)

                    # Show streaming images
                    cv2.imshow('What the model sees', auto)
                    cv2.imshow('Original', image)

                    # Neural network model makes prediciton
                    prediction = model.predict(auto)

                    # Send prediction to driver to tell it how to steer
                    driver.steer(prediction)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed (video)"



class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread('192.168.1.66', 8000))
    video_thread.start()


if __name__ == '__main__':

    print '\n Hold on to your butts! \n'
    ThreadServer()
