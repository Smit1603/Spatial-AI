import cv2
import depthai as dai
import numpy as np
import time
import torch
import sys
from scipy.spatial import distance,cKDTree
from PIL import Image as im
import os


def fusion(Wc,Zs,Zm):
    Wc=1/(1+np.exp(0.25*cv2.normalize(np.float32(Wc),None,0,5,norm_type=cv2.NORM_MINMAX)))
    #Wc=np.where(Wc>0.5,1,0)
    Zs=Zs+1e-7
    Zm=Zm+1e-7
   
    # Zs=cv2.cvtColor(Zs,cv2.COLOR_RGB2GRAY)+1e-7
    # cv2.imshow("Zs",Zs)
    # cv2.waitKey(1)
    # Zm=cv2.cvtColor(Zm,cv2.COLOR_BGR2GRAY)+1e-7
    maxZs=np.max(Zs)
    maxZm_scale=255
    ratio=maxZs/maxZm_scale
    
    # print(Zm.shape)
    # print(Zs.shape)
    # print(orgimg.shape)
    #print(maxZs,maxZm)
    # Wc=get_Wc(orgimg)
    # Wc=(Wc*255).astype(np.uint8)
    # cv2.imshow("wc",Wc)
    # cv2.waitKey(1)
    # print(Wc)
    Nzm=Zm/maxZm_scale
    Nzs=Zs/maxZs
    Ws=np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm)
    #Ws=np.exp(np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm))/2.71
   # Z_final=(Wc)*Zs+(1-Wc)*(Zm*ratio)
    Z_final=(Wc)*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs)
    #Z_final=np.where(Wc<0.23,Zm*ratio,Wc*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs))
    #print(np.max(Z_final))
    # print(Z_final)
    return Z_final
blobfolder=os.path.dirname(os.path.abspath(__file__))+"/Converter/Blob"
try:
    blobfilename=os.listdir(blobfolder)[0]
    
    blobpath=blobfolder+"/"+blobfilename

    if not os.path.isfile(blobpath):
        blobpath=blobpath+"/"+os.listdir(blobpath)[0]
    
    if os.stat(blobpath).st_size<45600000:
        shape = (1, 256, 256)
    else:
        shape=(1,384,384)
except: 
    raise Exception("Blob files have not been correctly stored inside blobs folder")




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

# if (model_type=="midas-small"):
#     blobpath="/home/raghav/depthai-python/examples/ColorCamera/model-small-simplified-final_6.blob/model-small-simplified_openvino_2021.4_6shave.blob"
#     shape = (1, 256, 256)
# elif (model_type=="midas-hybrid"):
#     blobpath="/home/raghav/depthai-python/examples/ColorCamera/dpt_hybrid-simplified_openvino_2021.4_6shave_2.blob"
#     shape=(1,384,384)



# Create pipeline
device = dai.Device()
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
#depth.setInputResolution(1920,1200)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setDepthAlign(dai.CameraBoardSocket.RGB)
#depth.setOutputSize(256,256)








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
#camRgb.setVideoSize(480, 640)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(60)


#defining image manips 
manipRgb = pipeline.create(dai.node.ImageManip)
manipRgb2 = pipeline.create(dai.node.ImageManip)
manipStereo = pipeline.create(dai.node.ImageManip)
manipConf = pipeline.create(dai.node.ImageManip)
# manipLeft= pipeline.create(dai.node.ImageManip)
# manipRight=pipeline.create(dai.node.ImageManip)
# manipLeft2= pipeline.create(dai.node.ImageManip)
# manipRight2=pipeline.create(dai.node.ImageManip)

#resizing images using image manip
manipRgb.initialConfig.setResize(*(shape[1:]))
manipRgb2.initialConfig.setResize(*(shape[1:]))


manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

manipRgb.setKeepAspectRatio(1)
manipRgb2.setKeepAspectRatio(1)
#manipRgb.setNumFramesPool(60)

manipStereo.initialConfig.setResize(*(shape[1:]))
manipConf.initialConfig.setResize(*(shape[1:]))
# manipLeft.initialConfig.setResize(288,300)
# manipRight.initialConfig.setResize(288,300)
#manipLeft.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
#manipLeft.setNumFramesPool(60)


# manipLeft2.initialConfig.setResize(640, 400)
# manipRight2.initialConfig.setResize(640, 400)

#manipRight.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
#manipRight.setNumFramesPool(60)


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

# RGB_xout = pipeline.create(dai.node.XLinkOut)
# RGB_xout.setStreamName("RGB")

Confidence_xout=pipeline.create(dai.node.XLinkOut)
Confidence_xout.setStreamName("CMAP")

# Stereo_xout.input.setBlocking(True)
# Stereo_xout.input.setQueueSize(4)

#linking -->


# monoLeft.out.link(manipLeft2.inputImage)
# monoRight.out.link(manipRight2.inputImage)

# manipLeft.out.link(manipLeft2.inputImage)
# manipRight.out.link(manipRight2.inputImage)


# monoLeft.out.link(manipLeft2.inputImage)
# monoRight.out.link(manipRight2.inputImage)


monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)






# manipRgb2.out.link(RGB_xout.input)
#manipRgb.out.link(nn_xout.input)
nn.out.link(nn_xout.input)


depth.disparity.link(manipStereo.inputImage)
depth.confidenceMap.link(manipConf.inputImage)
manipConf.out.link(Confidence_xout.input)

#depth.disparity.link(Stereo_xout.input)

manipStereo.out.link(Stereo_xout.input)


# xoutVideo = pipeline.create(dai.node.XLinkOut)

# xoutVideo.setStreamName("video")

# Connect to device and start pipeline




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
        # frame_count+=1
        
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
            # print(frameConf)
            cv2.imshow("CMAP", frameConf)

        if latestPacket["STEREO"] is not None:
            frameDisp = latestPacket["STEREO"].getFrame()
            maxDisparity = depth.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frameDispColor = cv2.applyColorMap(frameDisp, cv2.COLORMAP_MAGMA)
            frameDispColor = np.ascontiguousarray(frameDispColor)
            cv2.imshow("STEREO", frameDispColor)


        if latestPacket["RGB_MIDAS_VIDEO"] is not None:
            frameMidas=get_frame(latestPacket["RGB_MIDAS_VIDEO"],shape)
            frameMidas=cv2.normalize(np.float32(frameMidas),None,0,1,norm_type=cv2.NORM_MINMAX)
            frameMidas=((frameMidas*255).astype(np.uint8))
            frameMidasColor=cv2.applyColorMap(frameMidas,cv2.COLORMAP_MAGMA)
            cv2.imshow('MIDAS-RGB-OAKD',frameMidasColor)
        
        if frameConf is not None and frameDisp is not None and frameMidas is not None:
            frame_count+=1
            fused_frame=fusion(frameConf,frameDisp,frameMidas)
            fused_frame=cv2.normalize(np.float32(fused_frame),None,0,1,norm_type=cv2.NORM_MINMAX)
            fused_frame = cv2.medianBlur(fused_frame, 5)
            fused_frame=(fused_frame*255).astype(np.uint8)
            fused_frame=cv2.applyColorMap(fused_frame,cv2.COLORMAP_MAGMA)
            cv2.imshow('FUSION',fused_frame)
            frameMidas,frameDisp,frameConf=None,None,None
            if frame_count%20==0:
                new_frame_time = time.time()
                fps = 1 / ((new_frame_time - prev_frame_time)/20)
                prev_frame_time=new_frame_time
                fps = str(int(fps))
                print("FPS : {} ".format(fps))
       
        

        if cv2.waitKey(1) == ord('q'):
            break