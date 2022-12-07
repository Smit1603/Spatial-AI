
import cv2
import depthai as dai
import numpy as np
import time
import torch
import sys


def getFrame(queue):
    frame = queue.get()
   
    return frame.getCvFrame()


model_type='midas-small'

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

if (model_type=="midas-small"):
    blobpath="/home/raghav/depthai-python/examples/ColorCamera/model-small-simplified-final_6.blob/model-small-simplified_openvino_2021.4_6shave.blob"
    shape = (1, 256, 256)
elif (model_type=="midas-hybrid"):
    blobpath="/home/raghav/depthai-python/examples/ColorCamera/dpt_hybrid-simplified_openvino_2021.4_6shave_2.blob"
    shape=(1,384,384)



# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)


monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
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


# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P )
camRgb.setVideoSize(480, 640)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(60)


#defining image manips 
manipRgb = pipeline.create(dai.node.ImageManip)
manipStereo = pipeline.create(dai.node.ImageManip)
manipLeft= pipeline.create(dai.node.ImageManip)
manipRight=pipeline.create(dai.node.ImageManip)
manipLeft2= pipeline.create(dai.node.ImageManip)
manipRight2=pipeline.create(dai.node.ImageManip)

#resizing images using image manip
manipRgb.initialConfig.setResize(*(shape[1:]))
manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manipRgb.setNumFramesPool(60)

manipStereo.initialConfig.setResize(*(shape[1:]))
manipLeft.initialConfig.setResize(288,300)
manipRight.initialConfig.setResize(288,300)


manipLeft2.initialConfig.setResize(640, 400)
manipRight2.initialConfig.setResize(640, 400)




camRgb.video.link(manipRgb.inputImage)

#create networks
nn = pipeline.create(dai.node.NeuralNetwork)

nn.setBlobPath(blobpath)
nn.input.setBlocking(True)

nnL = pipeline.create(dai.node.NeuralNetwork)
nnL.setBlobPath("/home/raghav/Blur5.blob")

nnR=pipeline.create(dai.node.NeuralNetwork)
nnR.setBlobPath("/home/raghav/Blur5.blob")



manipRgb.out.link(nn.input)

nn_xout = pipeline.create(dai.node.XLinkOut)
nn_xout.setStreamName("RGB_MIDAS_VIDEO")
nn_xout.input.setBlocking(True)
nn_xout.input.setQueueSize(1000)


Stereo_xout = pipeline.create(dai.node.XLinkOut)
Stereo_xout.setStreamName("Stereo")


#linking -->
monoLeft.out.link(manipLeft.inputImage)
monoRight.out.link(manipRight.inputImage)

manipLeft.out.link(nnL.input)
manipRight.out.link(nnR.input)

nnL.passthrough.link(manipLeft2.inputImage)
nnR.passthrough.link(manipRight2.inputImage)
manipLeft2.out.link(depth.left)
manipRight2.out.link(depth.right)


nn.out.link(nn_xout.input)
depth.disparity.link(manipStereo.inputImage)
manipStereo.out.link(Stereo_xout.input)





def get_frame(imfFrame, shape):
    return (np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0))

counter = 0
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(100) 
    device.setIrFloodLightBrightness(0) 
    startTime = time.monotonic()
    video = device.getOutputQueue(name="RGB_MIDAS_VIDEO", maxSize=1000, blocking=False)
    q = device.getOutputQueue(name="Stereo", maxSize=10, blocking=False)
    while True:
        
        videoIn = video.tryGet()
        if videoIn!=None:
            frame=get_frame(videoIn,shape)
    
            frame=cv2.normalize(np.float32(frame),None,0,1,norm_type=cv2.NORM_MINMAX)
           
            frame=((frame*255).astype(np.uint8))
            
           
            frame=cv2.applyColorMap(frame,cv2.COLORMAP_MAGMA)
 
            cv2.imshow('MIDAS-RGB-OAKD',frame)
      
        inDisparity = q.get()  
        frame2 = inDisparity.getFrame()
     
        frame2 = (frame2 * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        frame2 = cv2.applyColorMap(frame2, cv2.COLORMAP_MAGMA)
        cv2.imshow("disparity_color", frame2)

        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
            print("FPS : {} ".format(fps))
       
        

        if cv2.waitKey(1) == ord('q'):
         
            break
        
        

