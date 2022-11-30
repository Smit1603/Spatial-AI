#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import torch
import sys




model_type='midas-small'

def get_frame_on_computer(frame1):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(frame1).to(device2)
    print(np.asarray(input_batch).shape)
    with torch.no_grad():
        prediction = midas(input_batch)
        #print(prediction.shape)
        # print(frame.shape[:2])
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame1.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    #print(prediction.shape)
    frame1 = prediction.cpu().numpy()
    #print(frame)
    # frame1=cv2.normalize(frame1,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    # frame1=(frame1*255).astype(np.uint8)
    # frame1=cv2.applyColorMap(frame1,cv2.COLORMAP_MAGMA)
    return frame1
def save_array(filename,Array):
    file = open(filename, "w")
    content = str(Array)
    file.write(content)
    file.close()
#blob path of midas dataset 
prev_frame_time,new_frame_time,frame_count=0,0,0
if (model_type=="midas-small"):
    blobpath="/home/raghav/depthai-python/examples/ColorCamera/model-small-simplified-final_8.blob/model-small-simplified_openvino_2021.4_6shave.blob"
    shape = (1, 256, 256)
elif (model_type=="midas-hybrid"):
    blobpath="/home/raghav/depthai-python/examples/ColorCamera/dpt_hybrid-simplified_openvino_2021.4_6shave_2.blob"
    shape=(1,384,384)

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)


# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P )
camRgb.setVideoSize(480, 640)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(30)


#defining image manips 
manipRgb = pipeline.create(dai.node.ImageManip)


#resizing images using image manip
manipRgb.initialConfig.setResize(*(shape[1:]))
manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
# Linking
camRgb.video.link(manipRgb.inputImage)


nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blobpath)
manipRgb.out.link(nn.input)

nn_xout = pipeline.create(dai.node.XLinkOut)
nn_xout.setStreamName("RGB_MIDAS_VIDEO")
nn_xout.input.setBlocking(False)
nn_xout.input.setQueueSize(60)

nn.out.link(nn_xout.input)

# xoutVideo = pipeline.create(dai.node.XLinkOut)

# xoutVideo.setStreamName("video")

# Connect to device and start pipeline




def get_frame(imfFrame, shape):
    return (np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0))
   
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="RGB_MIDAS_VIDEO", maxSize=60, blocking=True)
   
    while True:
        frame_count+=1
        videoIn = video.get()
        frame=get_frame(videoIn,shape)
        #print(frame)
        
       
        frame=cv2.normalize(np.float32(frame),None,0,1,norm_type=cv2.NORM_MINMAX)
      
        frame=((frame*255).astype(np.uint8))
      
        #print(frame)
        #print(frame.shape)
        frame=cv2.applyColorMap(frame,cv2.COLORMAP_MAGMA)
        # frame2=cv2.normalize(frame2,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
        
        # frame2=((frame2*255).astype(np.uint8))
      
        #print(frame)
        #print(frame.shape)
        #frame2=cv2.applyColorMap(frame2,cv2.COLORMAP_MAGMA)
        #frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #print(frame)
        
        cv2.imshow('MIDAS-RGB-OAKD',frame)
        #cv2.imshow('MIDAS-RGB-COMPUTER',frame2)
        #cv2.imshow('Difference',frame2-frame)
     
        #print(avg_difference)
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        
        if (frame_count%20==0):
            new_frame_time = time.time()
            fps = 1 / ((new_frame_time - prev_frame_time)/20)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            print("FPS : {} ".format(fps))
       
        

        if cv2.waitKey(1) == ord('q'):
         
            break
        
        

