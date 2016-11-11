# galvanEye
  Building on the original work of [Hamuchiwa](https://github.com/hamuchiwa/AutoRCCar), I incorporated image preprocessing in OpenCV and used Keras (TensorFlow backend) to train a neural network that could drive a remote control (RC) car and detect common environmental variables using computer vision. This project fulfilled the capstone requirement for my graduation from the Data Science Immersive program at Galvanize in Austin, Texas (August-November 2016). For a high-level overview of this project, please see this [slide deck](https://github.com/pseudoyim/galvaneye/blob/master/galvaneye.pptx).

## Motivation
  I've been following developments in the field of autonomous vehicles for several years now, and I'm very interested in the impacts these developments will have on public policy and in our daily lives. I wanted to learn more about the underlying machine learning techniques that make autonomous driving possible. Lacking access and resources to work with actual self-driving cars, I was happy to find that it was possible to work with an RC model, and I'm very grateful to Hamuchiwa for having demonstrated these possibilities through his own self-driving RC car project.

## The car
<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/9688260/20187772/41067a5a-a73b-11e6-95be-035bb971835f.jpg" width="700"/>
</p>

## Primary Python packages used
  - OpenCV v3.1.0
  - Keras (TensorFlow backend)
  - Numpy
  - Scikit-learn

## Code
- computer/ 
  -	augmented/ 
    - stores augmented images when running ***img_augment.py***
  -	cascade_xml/ 
    - trained cascade xml files for stop sign detection
  -	images/ 
    - folders containing frames collected on each data collection run
  -	logs/ 
    - recorded logs of each data collection run
  -	nn_h5/ 
    - saved model weights and architecture (h5 file format used in Keras)
  -	notebooks/ 
    - Jupyter Notebook files where I tested out various code
  -	test_frames_SAVED/ 
    - saved frames from each test run where the car drove itself
  -	test_frames_temp/ 
    - temp location before in-progress test frames are moved to ***test_frames_SAVED/***
  -	training_data/ 
    - training image data for neural network in npz format
  - ***autonomous.py***: Drives the car and detects stop signs and pedestrians.
  - ***car.py***: The library that sends driving commands to the Arduino.
  - ***img_augment.py***: Augments images (shifts and jiggles images) to create additional training samples. Works in conjunction with ***img_filter_multiply.py***, which then flips and doubles the size of the training dataset.
  - ***img_collect.py***: Run this to collect training data images.
  - ***img_collect_negatives.py***: Run this to collect full image frames to use as negative samples when training the Haar Cascade.
  - ***img_filter_multiply.py***: Runs in conjunction with ***img_augment.py*** to apply Canny filter, flip, and double training set images.
  - ***nn_training.py***: Run this to train the neural network.
  - ***nn_training_conv.py***: Run this to train a neural network with convolutional layers. (I ran this on AWS EC2 instance because it took a very long time. In the end, convultional layers did not help my model).
  
- notes/ 
  - Contains notes on how to run configurations for Raspberry Pi and OpenCV functions. The OpenCV functions are not very user-friendly, especially the steps required for creating sample images and training the Haar Cascade .xml file. I performed the Haar Cascade training on an AWS EC2 instance so that it would run faster and allow me to keep working on my laptop.

- raspberry_pi/ 
  -	***stream_client.py***: Unchanged from Hamuchiwa's version ("If it ain't broke, don't fix it"). Streams video frames in jpeg format from the Pi (client) to the host (computer).

- test/ 
  - Python scripts to test various components of this project, including:
    - controlling car with Python programming
    - controlling car manually using arrow keys
    - ***stream_server_test.py*** (Hamuchiwa's original version) to test the socket connection between Raspberry Pi and computer.

## Data collection
  I had to collect my own image data to train the neural network. This was a bit of a laborious task, as it involved:
  1. Measuring out a "test track" in my apartment and marking the lanes with masking tape. The turns of the track were dictated by the turning radius of the RC car, which, in my case, was not small.
  2. Manually driving the car around the track, a few inches at a time. Each time I pressed an arrow key, the car moved in that direction and it captured an image of the road in front of it, along with the direction I told it to move at that instance. I collected over 5,000 data points in this manner, which took about ten hours over the course of three days. Fortunately, after running the ***img_augment.py*** and ***img_filter_multiply.py*** scripts I had written, I had over 30,000 training images to work with.

## Training the neural network
  I used Keras (TensorFlow backend). Following Hamuchiwa's example, I kept the structure simple, with only one hidden layer. After training my best model, I was able to get an accuracy of about 81% on cross-validation. This model was used to have the car drive itself. On average, the car makes about one mistake per lap. In this context, a "mistake" could be defined as the car driving outside of the lanes with no hope of being able to find its way back.
  I attempted to add convolutional layers to the model to see if that would increase accuracy. In the end, these attempts did not pan out and I never got an accuracy above 50% using convolution.

## Tying it all together
  After training my first model, I began to feed it image frames on my laptop to see what kind of predictions it made. It was very exciting to see it output accurate directions given various frames of the track ("Left"==[1,0,0]; "Right"==[0,1,0]; "Forward"==[0,0,1]):
<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/9688260/20226904/a06f25b2-a80f-11e6-85b0-a5df06bf2887.png" width="700"/>
</p>
  The moment of truth came when I implemented the code for the trained model into the script that drove the car. A couple seconds after hitting "enter", I heard the car begin to move by itself. At that moment, I felt like:

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/9688260/20187773/4232f516-a73b-11e6-8782-067438af4388.jpg" width="500"/>
</p>
  
## Future goals
  Watching the car drive itself around the track is pretty amazing, but the mistakes it makes are fascinating in their own way. I'm interested in experimenting with ***reinforcement learning*** techniques that could potentially help the car get out of mistakes and find its way back onto the track by itself.
