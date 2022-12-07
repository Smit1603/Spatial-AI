# import torch.onnx
# import cv2
# import time
# import torch
# import numpy as np

# #model_type = "DPT_Hybrid"

# model_type = "DPT_Hybrid"
# model = torch.hub.load("intel-isl/MiDaS", model_type)
# # print(torch.version.cuda)
# # print(torch._version_)
# # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # print(torch.cuda.is_available())
# # print(device)
# # midas.to(device)
# # midas.eval()




# # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# # if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
# #     transform = midas_transforms.dpt_transform
# # else:
# #     transform = midas_transforms.small_transform


# # # Creating a VideoCapture object to read the video
# # cap = cv2.VideoCapture(0)
# # new_frame_time,prev_frame_time=0,0
# # # Loop until the end of the video
# # while (cap.isOpened()):

# #     # Capture frame-by-frame
# #     ret, frame = cap.read()


# #     # MIDAS COMPUTATIONS


# #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     input_batch = transform(frame).to(device)

# #     with torch.no_grad():
# #         prediction = midas(input_batch)

# #         prediction = torch.nn.functional.interpolate(
# #             prediction.unsqueeze(1),
# #             size=frame.shape[:2],
# #             mode="bicubic",
# #             align_corners=False,
# #         ).squeeze()

# #     frame = prediction.cpu().numpy()
# #     frame=cv2.normalize(frame,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
# #     frame=(frame*255).astype(np.uint😎
# #     frame=cv2.applyColorMap(frame,cv2.COLORMAP_MAGMA)


# #     new_frame_time = time.time()
# #     fps = 1 / (new_frame_time - prev_frame_time)
# #     prev_frame_time = new_frame_time
# #     fps = int(fps)
# #     fps = str(fps)

# #     cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

# #     cv2.imshow('MIDAS SMALL OUTPUT WITH FPS', frame)
# #     # define q as the exit button
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # release the video capture object
# # cap.release()
# # # Closes all the windows currently opened.
# # cv2.destroyAllWindows() 

# #Function to Convert to ONNX 
# def Convert_ONNX(): 

#     # set the model to inference mode 
#     model.eval() 


#     # Export the model   
#     torch.onnx.export(model,torch.zeros((1,3,256,256)),'dpt_hybrid-midas.pt') 

#     print(" ") 
#     print('Model has been converted to ONNX') 

# if __name__ == "__main__": 

#     # Let's build our model 
#     #train(5) 
#     #print('Finished Training') 

#     # Test which classes performed well 
#     #testAccuracy() 

#     # Let's load the model we just created and test the accuracy per label 
    
#     # path = "/home/raghav/depthai-python/examples/ColorCamera/dpt_hybrid-midas-501f0c75.pt" 
#     # model.load_state_dict(model) 

#     # Test with batch of images 
#     #testBatch() 
#     # Test how the classes performed 
#     #testClassess() 
#     Convert_ONNX() 
#     # Conversion to O
import cv2
import time
import torch
import numpy as np

#model_type = "DPT_Hybrid"

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
print(torch.version.cuda)
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(torch.cuda.is_available())
print(device)
midas.to(device)
midas.eval()




midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(0)
new_frame_time,prev_frame_time=0,0
# Loop until the end of the video
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()


    # MIDAS COMPUTATIONS


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(frame).to(device)
    print(np.asarray(input_batch).shape)
    with torch.no_grad():
        prediction = midas(input_batch)
        #print(prediction.shape)
        # print(frame.shape[:2])
        # prediction = torch.nn.functional.interpolate(
        #     prediction.unsqueeze(1),
        #     size=frame.shape[:2],
        #     mode="bicubic",
        #     align_corners=False,
        # ).squeeze()
    #print(prediction.shape)
    frame = prediction.cpu().numpy()
    print(frame.shape)
    #print(frame)
    frame=cv2.normalize(frame,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    frame=(frame*255).astype(np.uint8)
    frame=cv2.applyColorMap(frame,cv2.COLORMAP_MAGMA)


    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time=new_frame_time
    fps = int(fps)
    fps = str(fps)

    #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    print(fps)
    cv2.imshow('MIDAS SMALL OUTPUT WITH FPS', frame)
    # define q as the exit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()