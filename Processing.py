#!/usr/bin/env python3

import cv2
import os
import depthai as dai
import numpy as np
import time
current_time,startTime=0,0
counter=0
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True


blobfolder=os.path.dirname(os.path.abspath(__file__))+"/scripts/Blob"
shape=(1,256,256)
try:
    blobfilename=os.listdir(blobfolder)[0]
    
    blobpath=blobfolder+"/"+blobfilename

    if not os.path.isfile(blobpath):
        blobpath=blobpath+"/"+os.listdir(blobpath)[0]
    
except: 
    raise Exception("Blob files have not been correctly stored inside blobs folder")


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
manipRight = pipeline.create(dai.node.ImageManip)
manipLeft2 = pipeline.create(dai.node.ImageManip)
manipRight2 = pipeline.create(dai.node.ImageManip)
depth = pipeline.create(dai.node.StereoDepth)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)







config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)


monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


monoLeft.out.link(manipLeft.inputImage)
monoRight.out.link(manipRight.inputImage)


manipLeft.initialConfig.setResize(288, 300)
manipRight.initialConfig.setResize(288, 300)

manipLeft2.initialConfig.setResize(640, 400)
manipRight2.initialConfig.setResize(640, 400)

nnL = pipeline.create(dai.node.NeuralNetwork)
nnL.setBlobPath(blobpath)
manipLeft.out.link(nnL.input)

nnR = pipeline.create(dai.node.NeuralNetwork)
nnR.setBlobPath(blobpath)
manipRight.out.link(nnR.input)



nnL.passthrough.link(manipLeft2.inputImage)
nnR.passthrough.link(manipRight2.inputImage)
manipLeft2.out.link(depth.left)
manipRight2.out.link(depth.right)

# print("sj")

manipstereo = pipeline.create(dai.node.ImageManip)


manipstereo.initialConfig.setResize(*shape[1:])

xout = pipeline.create(dai.node.XLinkOut) 
xout.setStreamName("disparity")


depth.disparity.link(manipstereo.inputImage)
manipstereo.out.link(xout.input)


with dai.Device(pipeline) as device:


    q = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)



    while True:
        
        frame = getFrame(q)
       
      
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
      
       
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
            print("FPS : {} ".format(fps))
        
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)
        cv2.imshow("Stereo_Processed", frame)

        if cv2.waitKey(1) == ord('q'):
            break