
"""
Adapted from DeepLabCut Toolbox
by 
B Forys, brandon.forys@alumni.ubc.ca
P Gupta, pankajgupta@alumni.ubc.ca

https://github.com/AlexEMG/DeepLabCut
by
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes a streaming video from a local webcam based
on a trained network. You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos_streamLocal.py

"""

####################################################
# Dependencies
####################################################
import os.path
import sys
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
else:
	print('Please use Python3')
	exit(0)

# Get subfolder containing this code
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow/")
sys.path.append(subfolder + "Generating_a_Training_Set")

# Import config information from myconfig_stream.py
from myconfig_stream import Task, date, \
	trainingsFraction, resnet, snapshotindex, shuffle, \
	cropping, x1, x2, y1, y2, plotimages, downscale, \
	pics, t, avgs

# Deep-cut dependencies
import compileall
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input
import pickle
from skimage.util import img_as_ubyte
import skimage
import skimage.color
import pandas as pd
import numpy as np
import os
import io
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime

# Image processing dependencies
import cv2
from PIL import Image, ImageTk
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS

# Threading dependencies
from joblib import Parallel, delayed
import _thread
import threading
from copy import copy, deepcopy

# GPIO dependencies
from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioController, GpioException
from os import environ
from led_test import LEDTest

####################################################
# Functions for processing each frame and analyzing its pose
####################################################

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	RGB color; the keyword argument name must be a standard mpl colormap name.
	From DeepLabCut'''
	return plt.cm.get_cmap(name, n)

def getpose(image, cfg, outputs, outall=False):
	''' Adapted from DeeperCut, see pose-tensorflow folder.
	From DeepLabCut'''
	image_batch = data_to_input(skimage.color.gray2rgb(image))
	outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
	scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
	pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
	if outall:
		return scmap, locref, pose
	else:
		return pose

def frame_process(image, cfg, outputs, index):
	'''Estimates pose in each frame.'''
	pose = getpose(image, cfg, outputs)
	print('Pose ' + str(index) + ' saved!')
	PredicteData[index, :] = pose.flatten()


def frame_plot(frame, index):
	'''Plots movement data on frames and saves frames.'''
	tmp = deepcopy(frame)
	tmp_no_label = deepcopy(frame)
	# Plots average of all points, or all points separately, depending on config
	if avgs:
		img = cv2.circle(tmp, (int(x_avg), int(y_avg)), 6, colors[1],-1)
	else:
		for x_plt, y_plt, c in zip(x_range, y_range, colors):
			cv2.circle(tmp, (int(PredicteData[index, :][x_plt]), \
			int(PredicteData[index, :][y_plt])), 2, c,-1)
	cv2.imwrite(out_dir + '/' + 'frame{}.png'.format(str(index)), tmp)
	cv2.imwrite(out_no_label + '/' + 'no_label_frame{}.png'.format(str(index)), tmp_no_label)

def thresh(LED, reached, ind):
	'''Controls GPIO (LED) output based on calculated threshold.'''
	if reached:
		print("Reached!")
		LED_arr.append(ind)
		# Default: Connect to GPIO port 7 on the breakout board
		try:
			LED.set_gpio(7, True)
		except GpioException:
			print("Error switching light on!")
		time.sleep(0.02)
		try:
			LED.set_gpio(7, False)
		except GpioException:
			print("Error switching light off!")
		time.sleep(0.02)


####################################################
# Loading data, and defining model folder
####################################################

basefolder = '../pose-tensorflow/models/'  # for cfg file & ckpt!
modelfolder = (basefolder + Task + str(date) + '-trainset' +
			   str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))
cfg = load_config(modelfolder + '/test/' + "pose_cfg.yaml")

##################################################
# Load and setup CNN part detector
##################################################

# Check which snap shots are available and sort them by # iterations
Snapshots = np.array([
	fn.split('.')[0]
	for fn in os.listdir(modelfolder + '/train/')
	if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print(modelfolder)
print(Snapshots)

##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

# Get training iterations:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
	date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
pdindex = pd.MultiIndex.from_product(
	[[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
	names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

# Pre-allocates array for data (change the first value in np.zeros to record more than 10,000 frames)
PredicteData = np.zeros((10000, 3 * len(cfg['all_joints_names'])))
index = 0
x_range = list(range(0,(3 * len(cfg['all_joints_names'])),3))
y_range = list(range(1,(3 * len(cfg['all_joints_names'])),3))

# Sets colors for cv2 plotting
colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160)]

# Sets output directory. When running multiple trials with same parameters, add a number
# to the end of this directory to prevent it from being overwritten
out_dir = '2MINS_THRESH_X_18_UPPERLIM_50_LED_02_out_' + Task + '/'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

# Generates a second directory with just the unlabelled frames (useful if you want to run
# further analysis on the real-time frames)
out_no_label = 'out_no_label_' + Task + '/'
if not os.path.exists(out_no_label):
	os.makedirs(out_no_label)

# Initializes arrays that will store movement data and LED flash data
x_arr = []
y_arr = []
x_overall = []
y_overall = []
LED_arr = []

print("Starting to extract posture")
# Accept a single connection and make a file-like object out of it
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
start = time.time()

# Calls custom pyftdi class in led_test.py. mask is an 8-bit array defining
# which ports on your breakout board are inputs and outputs; by default, 0xFF
# (all ports are output ports) should work.
LED = LEDTest()
mask = 0xFF

try:
	while True:   
		# Opens connection to breakout board
		LED.open(mask)
		frame = vs.read()
		if cropping:
			frame = frame[y1:y2, x1:x2]
			w = x2-x1
			h = y2-y1
		else:
			w = 400.0
			h = 400.0

		# if downscale == True, downscales frame by half
		if downscale:
			frame = imutils.resize(frame, width=400)

		# Break program if no more frames are received from webcam
		if frame is None:
			break

		if frame.any():
			image = (img_as_ubyte(frame))
			# Predict movement data from frame
			frame_process(frame, cfg, outputs, index)

			# Plot predicted movement data on frame
			if plotimages:
				# frame_plot(frame, index)
				_thread.start_new_thread(frame_plot, (frame, index))
				#threading.Thread(target=frame_plot, args=(frame, index))

		# Maps recorded movement data for each frame to separate arrays for x
		# and y coordinates
		for a in PredicteData[index, 0:14:3]:
			x_arr.append(a)
		for b in PredicteData[index, 1:15:3]:
			y_arr.append(b)

		x_avg = np.mean(x_arr)
		y_avg = np.mean(y_arr)

		# get first body part only
		# Change body part here:
		x_first = PredicteData[index, 9]
		y_first = PredicteData[index, 10]

		# update the FPS counter
		fps.update()

		# calculate measures of central tendency and variance
		x_avg = np.mean(x_arr)
		y_avg = np.mean(y_arr)
		x_overall.append(x_avg)
		y_overall.append(y_avg)
		#x_stdev = np.std(x_overall)
		#y_stdev = np.std(y_overall)
		x_stdev = 18 # right_paw: 14
		y_stdev = 18 # right_paw: 14
		x_upper_lim = 50
		y_upper_lim = 50

		# check whether movement in x or y direction exceeds threshold
		# and send signal to activate LED if threshold exceeded
		if(x_upper_lim >= abs(x_overall[index] - x_overall[index - 1]) >= x_stdev and \
		#abs(y_overall[index] - y_overall[index - 1]) >= y_stdev and \
		index >= 1):
			ttt = True
			_thread.start_new_thread(thresh, (LED, ttt, index))

		x_arr = []
		y_arr = []

		x_first = 0
		y_first = 0

		index += 1
		if time.time() - start > 120:
			break

# Finish run early if Ctrl+C pressed
except KeyboardInterrupt:
	print("Finished.")

finally:
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	LED.close()
	# server_socket.close()
	stop = time.time()
	stop_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
	dictionary = {
		"start": start,
		"stop": stop,
		"run_duration": stop - start,
		"Scorer": scorer,
		"config file": cfg,
		"frame_dimensions": (w, h),
		"nframes": index
	}
	metadata = {'data': dictionary}
	print("Frame rate: {} Hz".format(str(index / (stop - start))))
	print("Saving results...")
	PredicteData = PredicteData[0:index, :]
	DataMachine = pd.DataFrame(
		PredicteData, columns=pdindex, index=range(index))
	AvgsMachine = pd.DataFrame(
	{'x': x_overall, 'y': y_overall}
	)
	LED_frames = pd.DataFrame(
	{'LED_frames': LED_arr})

	# Record raw data (DataMachine), averaged data (AvgsMachine), and frames on which
	# LED was activated (LED_frames) to csv
	DataMachine.to_csv(out_dir + t + "_LEDwebcam_{}_{}_Run".format(Task, pics) + str(stop_time) + '.csv')
	AvgsMachine.to_csv(out_dir + t + "_LEDwebcam_{}_{}_Run_avgs_".format(Task, pics) + str(stop_time) + '.csv')
	LED_frames.to_csv(out_dir + t + "_LEDwebcam_{}_{}_LED_frames_".format(Task, pics) + str(stop_time) + '.csv')

	print("Results saved!")
