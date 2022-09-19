#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
camRgb = pipeline.create(dai.node.ColorCamera)
depth_normal = pipeline.create(dai.node.StereoDepth)
depth = pipeline.create(dai.node.StereoDepth)
xout_normal = pipeline.create(dai.node.XLinkOut)
xout = pipeline.create(dai.node.XLinkOut)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xout_normal.setStreamName("disparity_normal")
xout.setStreamName("disparity")
xoutRgb.setStreamName("rgb")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth_normal.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 3
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 200
config.postProcessing.thresholdFilter.maxRange = 20000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Linking
monoLeft.out.link(depth_normal.left)
monoRight.out.link(depth_normal.right)
depth_normal.disparity.link(xout_normal.input)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)
camRgb.preview.link(xoutRgb.input)

i=0
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

	# Output queue will be used to get the disparity frames from the outputs defined above
	q_normal = device.getOutputQueue(name="disparity_normal", maxSize=4, blocking=False)
	q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
	qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

	while True:
		inDisparity_normal = q_normal.get()
		inDisparity = q.get()  # blocking call, will wait until a new data has arrived

		frame_normal = inDisparity_normal.getFrame()
		frame = inDisparity.getFrame()
		inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

		# Normalization for better visualization
		frame_normal = (frame_normal * (255 / depth_normal.initialConfig.getMaxDisparity())).astype(np.uint8)

		# Normalization for better visualization
		frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

		# Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
		frame_normal = cv2.applyColorMap(frame_normal, cv2.COLORMAP_JET)
		cv2.imshow("disparity_color_normal", frame_normal)

		# Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
		frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
		cv2.imshow("disparity_color", frame)

		# Retrieve 'bgr' (opencv format) frame
		cv2.imshow("rgb", inRgb.getCvFrame())

		if cv2.waitKey(1) == ord('s'):

			cv2.imwrite('/home/prachi/Desktop/OAK-D/depthai/Spatial-AI/Normal_ColorDisparity_Map' + str(i) + '.png',frame_normal)

			cv2.imwrite('/home/prachi/Desktop/OAK-D/depthai/Spatial-AI/PostProcessing_ColorDisparity_Map' + str(i) + '.png',frame)

			cv2.imwrite('/home/prachi/Desktop/OAK-D/depthai/Spatial-AI/PostProcessing_RGB_Image' + str(i) + '.png',inRgb.getCvFrame())
							
			i += 1

		if cv2.waitKey(1) == ord('q'):
			break