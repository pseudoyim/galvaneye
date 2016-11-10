This is to train classifiers for TRAFFIC SIGNS.

1. Find positive and negative images and save them to separate folders.
    Stop sign:    25 x 25 px    20 positive     400 negative  *Hamuchiwa already has a classifier .xml for this. Validate your version against his.
    20 mph:       25 x 50 px    20 positive     400 negative
    55 mph:       25 x 50 px    20 positive     400 negative
    Wrong Way:    50 x 25 px    20 positive     400 negative

2. Run the following from the command line to generate txt files with paths to each image.

    ## FROM DIRECTORY: Desktop/sign_stop (may need to move negatives.txt to the negatives folder)

    find positive_images -iname "*.jpeg" > positives/positives.txt
    find positives -iname "*.jpg" > positives/positives.txt
    find negatives -iname "*.jpg" > negatives/negatives.txt


3. 'createsamples' in cv2; the width and height specified are the dimensions for the positive image when superimposed on the background images.
   Make sure 'negatives.txt' is in the root folder.

    ## THIS WORKED!
    ## STOP SIGN (make sure the w & h specified below are for the size of the stop sign!)
    ## MOVE the 'negatives.txt' file out into the root directory.

    opencv_createsamples  -img positive_images/stop02.png \
                          -bg negatives.txt \
                          -info positives/positives.txt \
                          -pngoutput positives \
                          -num 800 \
                          -w 25 \
                          -h 25



4. Create 'positives.vec' file in the root directory.

    ## RUN THIS IMMEDIATELY AFTER RUNNING THE CODE IN STEP 3!

    opencv_createsamples  -info ./positives/positives.txt \
                          -num 800 \
                          -w 25 \
                          -h 25 \
                          -vec positives.vec



5. (see Hamuchiwa's advice on how to set parameters for training; trained pretty quickly) 'traincascade'; move 'negatives.txt' to the 'negatives' folder; make sure 'data' folder is empty. When finished, you'll have a cascade .xml file.


    ## Now copy all the relevant contents from the root directory to EC2, and train_cascade on a tmux.

    opencv_traincascade -data data -vec positives.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 800 -numNeg 2340 -featureType HAAR -w 25 -h 25 -mode ALL -precalcValBufSize 5120 -precalcIdxBufSize 5120
