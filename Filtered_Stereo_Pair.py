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

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
manipLeft = pipeline.create(dai.node.ImageManip)
manipRight= pipeline.create(dai.node.ImageManip)
# image_manip_script = pipeline.create(dai.node.Script)


monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


monoLeft.out.link(manipLeft.inputImage)
monoRight.out.link(manipRight.inputImage)


manipLeft.initialConfig.setResize(288, 300)
manipRight.initialConfig.setResize(288, 300)


# NN that detects faces in the image
nnL = pipeline.create(dai.node.NeuralNetwork)
nnL.setBlobPath("Blob/blur_simplified1_openvino_2021.4_6shave.blob")
manipLeft.out.link(nnL.input)

nnR = pipeline.create(dai.node.NeuralNetwork)
nnR.setBlobPath("Blob/blur_simplified1_openvino_2021.4_6shave.blob")
manipRight.out.link(nnR.input)

nn_xoutR = pipeline.create(dai.node.XLinkOut)
nn_xoutL = pipeline.create(dai.node.XLinkOut)

nnR.passthrough.link(nn_xoutR.input)
nnL.passthrough.link(nn_xoutL.input)

nn_xoutR.setStreamName("RIGHT_FRAME")
nn_xoutL.setStreamName("LEFT_FRAME")

with dai.Device(pipeline) as device:
    
    leftFrame = device.getOutputQueue(name="LEFT_FRAME", maxSize=10, blocking=False)
    rightFrame = device.getOutputQueue(name="RIGHT_FRAME", maxSize=10, blocking=False)
    
    while True:
        
        frameL = getFrame(leftFrame)
        frameR = getFrame(rightFrame)

        cv2.imshow("Right Frame " , frameR)
        cv2.imshow("Left Frame " , frameL)        
        
        if cv2.waitKey(1) == ord('q'):
            break