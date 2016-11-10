This is to train classifiers for TRAFFIC SIGNS.

1. Find positive and negative images and save them to separate folders.
    Stop sign:    25 x 25 px    20 positive     400 negative  *Hamuchiwa already has a classifier .xml for this. Validate your version against his.
    20 mph:       25 x 50 px    20 positive     400 negative
    55 mph:       25 x 50 px    20 positive     400 negative
    Wrong Way:    50 x 25 px    20 positive     400 negative

2. Run the following from the command line to generate txt files with paths to each image.

    find ./speed_20_pos -iname "*.png" > positives/positives.txt
    find ./speed_20_neg -iname "*.png" > negatives/negatives.txt


    ## FROM DIRECTORY: Desktop/sign_stop (may need to move negatives.txt to the negatives folder)
    find positive_images -iname "*.jpeg" > positives/positives.txt
    find negatives -iname "*.jpg" > negatives/negatives.txt


3. 'createsamples' in cv2; the width and height specified are the dimensions for the positive image when superimposed on the background images.
   Make sure 'negatives.txt' is in the root folder.

    opencv_createsamples  -img speed_20_pos/pos_1.png \
                          -bg negatives.txt \
                          -info positives/positives.txt \
                          -pngoutput positives \
                          -num 1000 \
                          -w 50 \
                          -h 62


    # To add a space between each line in negatives.txt file: (execute in ipython)
    >> filename = 'negatives.txt'
    >> f = open(filename)
    >> r = f.read()
    >> y = ''
    >> for x in r.split():
        y += x + '\n\n'
    >> b = open('negatives_new.txt', 'w')
    >> b.write(y)
    >> b.close()


    ## THIS WORKED!
    # STOP SIGN (make sure the w & h specified below are for the size of the stop sign!)
    opencv_createsamples  -img positive_images/stop_pos.png \
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

    # HAMUCHIWA'S
    opencv_traincascade -data data -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 800 -numNeg 400 -featureType HAAR -w 20 -h 20 -mode ALL -precalcValBufSize 5120 -precalcIdxBufSize 5120

    ## STOP sign
    opencv_traincascade -data data -vec positives.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 800 -numNeg 400 -featureType HAAR -w 25 -h 25 -mode ALL -precalcValBufSize 5120 -precalcIdxBufSize 5120

6.
