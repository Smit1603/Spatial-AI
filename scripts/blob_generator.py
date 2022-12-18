import onnx
from onnxsim import simplify
import blobconverter
def blob_generators(key):
    if(key==1):
        onnx_model = onnx.load("Onnx/Midas_Small.onnx")
        model_simpified, check = simplify(onnx_model)
        onnx.save(model_simpified, "Onnx/Midas_Small-simplified.onnx")

        blobconverter.from_onnx(
            model="Onnx/Midas_Small-simplified.onnx",
            output_dir="Blob/Midas-Small.blob",
            data_type="FP16",
            shaves=6,
            compile_params=["-ip U8"],
            use_cache=False,
            optimizer_params=["--data_type=FP16","--mean_values=[123.675, 116.280, 103.530]", "--scale_values=[58.395, 57.120, 57.375]"]
        )
    elif(key==2):
        onnx_model = onnx.load("Onnx/Midas_Hybrid.onnx")
        model_simpified, check = simplify(onnx_model)
        onnx.save(model_simpified, "Onnx/Midas_Hybrid-simplified.onnx")

        blobconverter.from_onnx(
            model="Onnx/Midas_Hybrid-simplified.onnx",
            output_dir="Blob/Midas-Hybrid.blob",
            data_type="FP16",
            shaves=6,
            compile_params=["-ip U8"],
            use_cache=False,
            optimizer_params=["--data_type=FP16","--mean_values=[127.5, 127.5, 127.5]", "--scale_values=[127.5,127.5,127.5]"]
        )        