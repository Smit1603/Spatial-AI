




import cv2
import depthai as dai
import numpy as np
import time
import torch
import sys
from scipy.spatial import distance,cKDTree
from PIL import Image as im
import os 
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

    


def fusion(orgimg,Zs,Zm):
    Zs=Zs+1e-7
    Zm=Zm+1e-7
    maxZs=np.max(Zs)
    maxZm_scale=255
    ratio=maxZs/maxZm_scale
    Wc=get_Wc(orgimg)
    Nzm=Zm/maxZm_scale
    Nzs=Zs/maxZs
    Ws=np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm)
    Z_final=np.where(Wc==0,Zm*ratio,Wc*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs))
    return Z_final
def get_Wc(frame):
    img=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    v = np.median(img)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    imgCanny=cv2.Canny(img,lower_thresh,upper_thresh)
    imgCanny=np.asarray(imgCanny)
    cv2.imshow("iCanny",imgCanny)
    cv2.waitKey(1)
   
    edges_coords=np.where(imgCanny==255)
    edges_coords=np.column_stack((edges_coords[0],edges_coords[1]))
   
    tree = cKDTree(edges_coords)
    
    coords=np.where(img>-1)
    coords=np.column_stack((coords[0],coords[1]))

    min_distance_arr=tree.query(coords,distance_upper_bound=5)[0].reshape(*shape[1:])
    wc= 1/(1+np.exp(0.25*min_distance_arr))
    return wc


def getFrame(queue):
    frame = queue.get()
   
    return frame.getCvFrame()





extended_disparity = False

subpixel = False

lr_check = True



new_frame_time,prev_frame_time,frame_count=0,0,0
# Create pipeline
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


#resizing images using image manip
manipRgb.initialConfig.setResize(*(shape[1:]))
manipRgb2.initialConfig.setResize(*(shape[1:]))


manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

manipRgb.setKeepAspectRatio(1)
manipRgb2.setKeepAspectRatio(1)


manipStereo.initialConfig.setResize(*(shape[1:]))






camRgb.isp.link(manipRgb.inputImage)
camRgb.isp.link(manipRgb2.inputImage)


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

RGB_xout = pipeline.create(dai.node.XLinkOut)
RGB_xout.setStreamName("RGB")


monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)






manipRgb2.out.link(RGB_xout.input)

nn.out.link(nn_xout.input)


depth.disparity.link(manipStereo.inputImage)



manipStereo.out.link(Stereo_xout.input)





def get_frame(imfFrame, shape):
    return (np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0))

counter = 0
with device:
    frameRgb=None
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
        latestPacket["RGB"] = None
        latestPacket["STEREO"] = None
        latestPacket["RGB_MIDAS_VIDEO"] = None

        queueEvents = device.getQueueEvents(("RGB", "STEREO","RGB_MIDAS_VIDEO"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]
        

        if latestPacket["RGB"] is not None:
            frameRgb = latestPacket["RGB"].getCvFrame()
            cv2.imshow("RGB", frameRgb)

        if latestPacket["STEREO"] is not None:
            frameDisp = latestPacket["STEREO"].getFrame()
            maxDisparity = depth.initialConfig.getMaxDisparity()
            
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
           
            if 1: frameDispColor = cv2.applyColorMap(frameDisp, cv2.COLORMAP_MAGMA)
            frameDispColor = np.ascontiguousarray(frameDispColor)
            cv2.imshow("STEREO", frameDispColor)


        if latestPacket["RGB_MIDAS_VIDEO"] is not None:
            frameMidas=get_frame(latestPacket["RGB_MIDAS_VIDEO"],shape)
            frameMidas=cv2.normalize(np.float32(frameMidas),None,0,1,norm_type=cv2.NORM_MINMAX)
            frameMidas=((frameMidas*255).astype(np.uint8))
            frameMidasColor=cv2.applyColorMap(frameMidas,cv2.COLORMAP_MAGMA)
            cv2.imshow('MIDAS-RGB-OAKD',frameMidasColor)
        
        if frameRgb is not None and frameDisp is not None and frameMidas is not None:
   
            fused_frame=fusion(frameRgb,frameDisp,frameMidas)
            fused_frame=cv2.normalize(np.float32(fused_frame),None,0,1,norm_type=cv2.NORM_MINMAX)
            fused_frame = cv2.medianBlur(fused_frame, 5)
            fused_frame=(fused_frame*255).astype(np.uint8)
           
            fused_frame=cv2.applyColorMap(fused_frame,cv2.COLORMAP_MAGMA)
            cv2.imshow('FUSION',fused_frame)
            try:
                if (not np.sum(np.abs(frameDisp-prevFrameDisp))<50):
                    frameMidas,frameDisp,frameRgb=None,None,None
            except:
                pass
            prevFrameDisp=frameDisp

        if cv2.waitKey(1) == ord('q'):
            break




        
      