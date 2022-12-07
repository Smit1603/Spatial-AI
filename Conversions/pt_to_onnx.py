

import cv2
import time
import torch
import numpy as np

from torch.autograd import Variable

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# midas.load_state_dict(torch.load('/home/raghav/Downloads/dpt_hybrid-midas-501f0c75.pt'))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 384, 384)) 
torch.onnx.export(midas, dummy_input, '/home/raghav/Downloads/dpt_hybrid_2.onnx',opset_version=11)
