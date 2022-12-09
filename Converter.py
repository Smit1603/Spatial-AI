#! /usr/bin/env python3
import cv2
from pathlib import Path
import tensorflow
import torch
from torch import nn
import kornia
import onnx
from onnxsim import simplify
import blobconverter
import torchvision
import numpy as np

name = 'blur'

class Model(nn.Module):
    def forward(self, image):
        frame = kornia.filters.gaussian_blur2d(image, (9, 9), (2.5, 2.5))
        return frame

# Define the expected input shape (dummy input)
shape = (1, 1, 300, 288)
model = Model()
X = torch.ones(shape, dtype=torch.float32)

path = Path("out/")
path.mkdir(parents=True, exist_ok=True)
onnx_path = str(path / (name + '.onnx'))
print(onnx_path)

print(f"Writing to {onnx_path}")
torch.onnx.export(
    model,
    X,
    onnx_path,
    opset_version=12,
    do_constant_folding=True,
)

onnx_simplified_path = str(path / (name + '_simplified1.onnx'))

# Use onnx-simplifier to simplify the onnx model
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, onnx_simplified_path)

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_simplified_path,
    data_type="FP16",
    shaves=6,
    use_cache=False,
    output_dir="Blob",
    optimizer_params=[]
)