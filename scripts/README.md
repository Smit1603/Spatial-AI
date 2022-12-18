# Scripts


## About

Above folder consists of all required files and codes required to convert pretrained models to onnx files so that they can be converted to blob files to be integrated on **Oak-D Pipeline**.

## Steps
#### 1. For generating blob path for Monocular Depth Estimation (MDE) 


To convert pretrained model to blob file , obey following steps :


1. Download weights of the model from given link.
   - [Midas Small](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt)
   - [Midas Hybrid](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt)

2. Store downloaded weights in folder `Weights`.
(And update the path of the weights in code)

3. Run the following script on your terminal window.
```
    python3 MDE.py
```

4. Check whether onnx model has been generated in `Onnx` folder.

5. Check whether blob model has been generated in `Blob` folder.

#### 2. For generating blob path for Pre-Processing


1. Run following commands on terminal prompt
```
    python3 Pre_Processing.py
```
