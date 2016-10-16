This is to train classifiers for TRAFFIC SIGNS.

1. Find positive and negative images and save them to separate folders.
    Stop sign:    25 x 25 px    20 positive     400 negative  *Hamuchiwa already has a classifier .xml for this. Validate your version against his.
    20 mph:       25 x 50 px    20 positive     400 negative
    55 mph:       25 x 50 px    20 positive     400 negative
    Wrong Way:    50 x 25 px    20 positive     400 negative

2. Run the following from the command line to generate txt files with paths to each image.

    find ./speed_20_pos -iname "*.png" > positives/positives.txt
    find ./speed_20_neg -iname "*.png" > negatives/negatives.txt


3. 'createsamples' in cv2; the width and height specified are the dimensions for the positive image when superimposed on the background images.
   Make sure 'negatives.txt' is in the root folder.

    opencv_createsamples  -img speed_20_pos/pos_1.png \
                          -bg negatives.txt \
                          -info positives/positives.txt \
                          -pngoutput positives \
                          -num 1000 \
                          -w 50 \
                          -h 62


4. Create 'positives.vec' file in the root directory.
    opencv_createsamples  -info positives/positives.txt \
                          -num 1000 \
                          -w 50 \
                          -h 62 \
                          -vec positives.vec


5. (see Hamuchiwa's advice on how to set parameters for training; trained pretty quickly) 'traincascade'; move 'negatives.txt' to the 'negatives' folder; make sure 'data' folder is empty. When finished, you'll have a cascade .xml file.

    opencv_traincascade -data data -vec positives.vec -bg negatives/negatives.txt -numPos 1000 -numNeg 5000 -numStages 5 -w 50 -h 62

    (next time) opencv_traincascade -data data -vec positives.vec -bg negatives/negatives.txt -numPos 1000 -numNeg 5000 -numStages 5 -w 50 -h 62 -acceptanceRatioBreakValue 10e-5



6.
