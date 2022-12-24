import cv2
import depthai as dai
import numpy as np
import time
import torch
import sys
import os 
from PIL import Image as im
import os


def fusion(Wc,Zs,Zm):
    Wc=1/(1+np.exp(0.25*cv2.normalize(np.float32(Wc),None,0,5,norm_type=cv2.NORM_MINMAX)))
    Zs=Zs+1e-7
    Zm=Zm+1e-7
    maxZs=np.max(Zs)
    maxZm_scale=255
    ratio=maxZs/maxZm_scale
    Nzm=Zm/maxZm_scale
    Nzs=Zs/maxZs
    Ws=np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm)
    Z_final=(Wc)*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs)
    return Z_final

blobfolder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/scripts/Blob"
blobfilename=os.listdir(blobfolder)[0]

blobpath=blobfolder+"/"+blobfilename

if not os.path.isfile(blobpath):
    blobpath=blobpath+"/"+os.listdir(blobpath)[0]
if blobfilename==".gitkeep":
    raise Exception("Blob file has not been correctly stored inside scripts/Blob folder")
if blobfilename=="Midas-Small.blob":
    shape=(1, 256, 256)
elif blobfilename=="Midas-Hybrid.blob":
    shape=(1,384,384)
elif os.stat(blobpath).st_size<45600000:
    shape = (1, 256, 256)
else:
    shape=(1,384,384)



def getFrame(queue):
    frame = queue.get()
   
    return frame.getCvFrame()


model_type='midas-small'
new_frame_time,prev_frame_time,frame_count=0,0,0
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True





device = dai.Device()
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)


camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setDepthAlign(dai.CameraBoardSocket.RGB)









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

camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(60)


#defining image manips 
manipRgb = pipeline.create(dai.node.ImageManip)
manipRgb2 = pipeline.create(dai.node.ImageManip)
manipStereo = pipeline.create(dai.node.ImageManip)
manipConf = pipeline.create(dai.node.ImageManip)


#resizing images using image manip
manipRgb.initialConfig.setResize(*(shape[1:]))
manipRgb2.initialConfig.setResize(*(shape[1:]))


manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

manipRgb.setKeepAspectRatio(1)
manipRgb2.setKeepAspectRatio(1)


manipStereo.initialConfig.setResize(*(shape[1:]))
manipConf.initialConfig.setResize(*(shape[1:]))




camRgb.isp.link(manipRgb.inputImage)
camRgb.isp.link(manipRgb2.inputImage)

#create networks
nn = pipeline.create(dai.node.NeuralNetwork)

nn.setBlobPath(blobpath)
nn.input.setBlocking(True)





manipRgb.out.link(nn.input)

nn_xout = pipeline.create(dai.node.XLinkOut)
nn_xout.setStreamName("RGB_MIDAS_VIDEO")
nn_xout.input.setBlocking(True)
nn_xout.input.setQueueSize(1000)


Stereo_xout = pipeline.create(dai.node.XLinkOut)
Stereo_xout.setStreamName("STEREO")



Confidence_xout=pipeline.create(dai.node.XLinkOut)
Confidence_xout.setStreamName("CMAP")






monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
nn.out.link(nn_xout.input)


depth.disparity.link(manipStereo.inputImage)
depth.confidenceMap.link(manipConf.inputImage)
manipConf.out.link(Confidence_xout.input)



manipStereo.out.link(Stereo_xout.input)






def get_frame(imfFrame, shape):
    return (np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0))

counter = 0
with device:
   
    frameConf=None
    frameDisp=None
    frameMidas=None
    device.startPipeline(pipeline)
    device.setIrLaserDotProjectorBrightness(100) # in mA, 0..1200
    device.setIrFloodLightBrightness(0) 
    camRgb.setIspScale(2, 3)
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise
    startTime = time.monotonic()
    
    while True:
       
        latestPacket = {}
        
        latestPacket["STEREO"] = None
        latestPacket["RGB_MIDAS_VIDEO"] = None
        latestPacket["CMAP"] = None
        queueEvents = device.getQueueEvents(("STEREO","RGB_MIDAS_VIDEO","CMAP"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]
        

        
        if latestPacket["CMAP"] is not None:
            
            frameConf = latestPacket["CMAP"].getCvFrame()
           

        if latestPacket["STEREO"] is not None:
            frameDisp = latestPacket["STEREO"].getFrame()
            maxDisparity = depth.initialConfig.getMaxDisparity()
         
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            
            if 1: frameDispColor = cv2.applyColorMap(frameDisp, cv2.COLORMAP_MAGMA)
            frameDispColor = np.ascontiguousarray(frameDispColor)
            



        if latestPacket["RGB_MIDAS_VIDEO"] is not None:
            frameMidas=get_frame(latestPacket["RGB_MIDAS_VIDEO"],shape)
            frameMidas=cv2.normalize(np.float32(frameMidas),None,0,1,norm_type=cv2.NORM_MINMAX)
            frameMidas=((frameMidas*255).astype(np.uint8))
            
        
        if frameConf is not None and frameDisp is not None and frameMidas is not None:
          
            fused_frame=fusion(frameConf,frameDisp,frameMidas)
            fused_frame=cv2.normalize(np.float32(fused_frame),None,0,1,norm_type=cv2.NORM_MINMAX)
            fused_frame = cv2.medianBlur(fused_frame, 5)
            fused_frame=(fused_frame*255).astype(np.uint8)
            fused_frame=cv2.applyColorMap(fused_frame,cv2.COLORMAP_MAGMA)
            cv2.imshow('FUSION',fused_frame)

            try:
                if (not np.sum(np.abs(frameDisp-prevFrameDisp))<50):
                    frameMidas,frameDisp,frameConf=None,None,None
            except:
                pass
            prevFrameDisp=frameDisp
            
           
       
        

        if cv2.waitKey(1) == ord('q'):
            break