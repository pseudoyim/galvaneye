## galvanEye

  Inspired by the original work of Hamuchiwa (https://github.com/hamuchiwa/AutoRCCar), this was the capstone project for my time in the August 2016 Data Science Immersive cohort at Galvanize in Austin, Texas. 
  
### Primary Python packages Used
* Computer:
  - OpenCV
  - Keras
  - Numpy
  
### Motivation
- raspberrt_pi/ 
  -	***stream_client.py***: (unchanged from Hamuchiwa's version) streams video frames in jpeg format to the host computer
interface
- computer/
  -	cascade_xml/ 
    - trained cascade classifiers xml files
  -	training_data/ 
    - training image data for neural network in npz format
  -	testing_data/ 
    - testing image data for neural network in npz format
  -	training_images/ 
    - saved video frames during image training data collection stage (optional)
  -	nn_h5/ 
    - trained neural network parameters in a xml file

### Data collection
1. **Layout of test track**: 
2. ** *“img_collect.py"* 
