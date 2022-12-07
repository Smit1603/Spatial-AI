import onnx
from onnxsim import simplify

onnx_model = onnx.load("/home/raghav/depthai-python/examples/ColorCamera/DenseDepth.onnx")
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, "/home/raghav/depthai-python/examples/ColorCamera/DenseDepth-simplified.onnx")

import blobconverter

blobconverter.from_onnx(
    model="/home/raghav/depthai-python/examples/ColorCamera/DenseDepth-simplified.onnx",
    output_dir="/home/raghav/depthai-python/examples/ColorCamera/DenseDepth.blob",
    data_type="FP16",
    shaves=6,
    compile_params=["-ip U8"],
    use_cache=False,
    optimizer_params=["--data_type=FP16"]
    #optimizer_params=["--data_type=FP16","--mean_values=[123.675, 116.280, 103.530]", "--scale_values=[58.395, 57.120, 57.375]"]
    #optimizer_params=["--data_type=FP16","--mean_values=[127.5, 127.5, 127.5]", "--scale_values=[127.5,127.5,127.5]"]
)
