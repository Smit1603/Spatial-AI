# Scripts


## About

Above folder consists of all required files and codes required to convert pretrained models to onnx files so that they can be converted to blob files to be integrated on **Oak-D Pipeline**.

## Steps
#### 1. For generating blob path for Monocular Depth Estimation (MDE) 


To convert pretrained model to blob file , obey following steps :

1. Run the following script on your terminal window in order to generate blob files for either Midas-small or Midas-Hybrid.

**Note** : Using Midas-Hybrid can reduce the fps by significant amounts 

```
    python3 MDE.py
```

2. Check whether onnx model has been generated in `Onnx` folder.

3. Check whether blob model has been generated in `Blob` folder.

Alternatively , if the above method does not work as expected you can directly download the blob files from the given link and place it in Blob folder  :

1. [Midas-Small](https://drive.google.com/file/d/1b-TD8QRocneNIceggphgY-qqx-quQpQs/view?usp=share_link)
2. [Midas-Hybrid](https://drive.google.com/file/d/12Y1ON640Ub1PhYEKWJ3HTVKBFAf9Exki/view?usp=share_link)


#### 2. For generating blob path for Pre-Processing


1. Run following commands on terminal prompt
```
    python3 Pre_Processing.py
```
Alternatively , if the above method does not work as expected you can directly download the blob file for pre-processing from [here](https://drive.google.com/file/d/1CTB0ICW1h2Z7RxcEv2zGRP0dTZBa_sFX/view?usp=share_link) and place it in Blob folder.

**Note** : It is recommended that you place only one blob file in blobs folder and try the corresponding method . For trying a new method you can generate a new blob file and replace it with previous .