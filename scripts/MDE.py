from blob_generator import blob_generators
from onnx_generator import onnx_generators

print("Enter 1 ---------->For Midas Small")
print("Enter 2 ---------->For Midas Hybrid")

key = int(input("Your Input : "))

onnx_generators(key)
blob_generators(key)