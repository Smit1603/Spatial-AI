#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# def getFrame(queue):
#     q=queue.get()
#     return q

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

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


manipLeft.initialConfig.setResize(300, 300)
manipRight.initialConfig.setResize(300, 300)


# NN that detects faces in the image
nnL = pipeline.create(dai.node.NeuralNetwork)
nnL.setBlobPath("/home/smit/Desktop/Spatial_AI_Image/Blob/blur_simplified1_openvino_2021.4_6shave.blob")
manipLeft.out.link(nnL.input)

nnR = pipeline.create(dai.node.NeuralNetwork)
nnR.setBlobPath("/home/smit/Desktop/Spatial_AI_Image/Blob/blur_simplified1_openvino_2021.4_6shave.blob")
manipRight.out.link(nnR.input)


# nn.out.link(image_manip_script.inputs['LEFT_IMAGE'])
# nn.out.link(image_manip_script.inputs['RIGHT_IMAGE'])
nn_xoutR = pipeline.create(dai.node.XLinkOut)
nn_xoutR.setStreamName("RIGHT_FRAME")

nn_xoutL = pipeline.create(dai.node.XLinkOut)
nn_xoutL.setStreamName("LEFT_FRAME")

nnR.out.link(nn_xoutR.input)
nnL.out.link(nn_xoutL.input)

# image_manip_script.setScript("""
    
#     while True:

#         output_L = node.io['LEFT_IMAGE'].get()
#         output_R = node.io['RIGHT_IMAGE'].get()

#         node.io['LEFT_FINAL'].send(output_L)
#         node.io['RIGHT_FINAL'].send(output_R)

    
    
# """)
# nn_xoutR = pipeline.create(dai.node.XLinkOut)
# nn_xoutR.setStreamName("RIGHT_FRAME")

# nn_xoutL = pipeline.create(dai.node.XLinkOut)
# nn_xoutL.setStreamName("LEFT_FRAME")

with dai.Device(pipeline) as device:
    
    leftFrame = device.getOutputQueue(name="LEFT_FRAME", maxSize=4, blocking=False)
    rightFrame = device.getOutputQueue(name="RIGHT_FRAME", maxSize=4, blocking=False)
    shape = (1, 300, 300)
    def get_frame(imfFrame, shape):
        return np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)

    while True:
        # Lframe = leftFrame.get()
        #Rframe = rightFrame.get()

        # cv2.imshow("Left Frame " , Lframe )
        cv2.imshow("Right Frame " , get_frame(rightFrame.get(),shape ))
        cv2.imshow("Left Frame " , get_frame(leftFrame.get(),shape ))        
        
        if cv2.waitKey(1) == ord('q'):
            break