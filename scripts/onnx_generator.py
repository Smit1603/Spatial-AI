import cv2
import time
import torch
import numpy as np
from torch.autograd import Variable

def onnx_generators(key):
    if(key==1):
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Export the trained model to ONNX
        dummy_input = Variable(torch.randn(1, 3, 256, 256)) 
        torch.onnx.export(midas, dummy_input, 'Onnx/Midas_Small.onnx',opset_version=11)
    elif(key==2):
        model_type = "DPT_Hybrid"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Export the trained model to ONNX
        dummy_input = Variable(torch.randn(1, 3, 384, 384)) 
        torch.onnx.export(midas, dummy_input, 'Onnx/Midas_Hybrid.onnx',opset_version=11)
