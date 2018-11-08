# coding: utf-8

############################
# This configuration file sets various parameters for running a trained model,
# that performed well on train/test set on videos
# Use this file to configure a model for analyzing a video stream
# Adapted from DeepLabCut Toolbox
# by 
# B Forys, brandon.forys@alumni.ubc.ca
# P Gupta, pankajgupta@alumni.ubc.ca
# https://github.com/AlexEMG/DeepLabCut
############################

# Analysis Network parameters:
# Define the network to be used to analyze the stream here

scorer = 'Rene'
Task = 'webcam_right_paw'
date = 'Oct12_18'
trainingsFraction = 0.95  # Fraction of labeled images used for training
resnet = 50
snapshotindex = -1
shuffle = 1

cropping = True

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the cropped region.

x1 = 280
x2 = 480

y1 = 100
y2 = 300

# For plotting:
trainingsiterations = 10000  # type the number listed in .pickle file
pcutoff = 0.1  # likelihood cutoff for body part in image

# output images with plotted data?
plotimages = True
if plotimages:
    pics = "Pics"
else:
    pics = "NoPics"

# downscale image to lower resolution? (you will typically get better results by cropping
# the frame instead)
downscale = False

# is this a test trial or a final trial (this just adds a "Test" or "Actual" label to your data
testing = False
if testing:
    t = "Test"
else:
    t = "Actual"
    
# Do you want to plot the average of your labeled points in this run?
avgs = False
