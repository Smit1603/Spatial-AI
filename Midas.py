import cv2
import depthai as dai
import numpy as np
import time
import os
new_frame_time=0
prev_frame_time=0
frame_count=0


blobfolder=os.path.dirname(os.path.abspath(__file__))+"/scripts/Blob"
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

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
#defining image manips 
manipRgb = pipeline.create(dai.node.ImageManip)
#resizing images using image manip
manipRgb.initialConfig.setResize(*shape[1:])
manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
# Linking
camRgb.video.link(manipRgb.inputImage)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blobpath)
manipRgb.out.link(nn.input)
nn_xout = pipeline.create(dai.node.XLinkOut)
nn_xout.setStreamName("RGB_MIDAS_VIDEO")
nn_xout.input.setBlocking(False)
nn_xout.input.setQueueSize(10)
nn.out.link(nn_xout.input)



def get_frame(imfFrame, shape):
    return np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0)
with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="RGB_MIDAS_VIDEO", maxSize=10, blocking=False)
    while True:
    
        
        videoIn = video.get()
        frame=get_frame(videoIn,shape)
        frame=cv2.normalize(np.float32(frame),None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
        frame=(frame*255).astype(np.uint8)
        
        frame=cv2.applyColorMap(frame,cv2.COLORMAP_MAGMA)
        cv2.imshow('MIDAS-RGB',frame)
     
      
        if cv2.waitKey(1) == ord('q'):
            break