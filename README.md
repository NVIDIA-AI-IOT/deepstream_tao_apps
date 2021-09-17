# Integrate TAO model with DeepStream SDK

- [Integrate TAO model with DeepStream SDK](#integrate-tao-model-with-deepstream-sdk)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Download](#download)
    - [1. Download Source Code with SSH or HTTPS](#1-download-source-code-with-ssh-or-https)
    - [2. Download Models](#2-download-models)
  - [Build](#build)
    - [1. Build TRT OSS Plugin](#1-build-trt-oss-plugin)
    - [2. Build Sample Application](#2-build-sample-application)
  - [Run](#run)
  - [Information for Customization](#information-for-customization)
    - [TAO Models](#tao-models)
    - [Label Files](#label-files)
    - [DeepStream configuration file](#deepstream-configuration-file)
    - [Model Outputs](#model-outputs)
      - [1~2. Yolov3 / YoloV4](#12-yolov3--yolov4)
      - [3~6. RetinaNet / DSSD / SSD/ FasterRCNN](#36-retinanet--dssd--ssd-fasterrcnn)
      - [7. PeopleSegNet](#7-peoplesegnet)
      - [8~9. UNET/PeopleSemSegNet](#89-unetpeoplesemsegnet)
      - [10. multi_task](#10-multi_task)
    - [TRT Plugins Requirements](#trt-plugins-requirements)
  - [FAQ](#faq)
    - [Measure The Inference Perf](#measure-the-inference-perf)
    - [About misc folder](#about-misc-folder)
  - [Known issues](#known-issues)

## Description

This repository provides a DeepStream sample application based on [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) to run eight TAO models (**Faster-RCNN** / **YoloV3** / **YoloV4** /**SSD** / **DSSD** / **RetinaNet**/ **PeopleSegNet**/ **UNET**/ **multi_task**/ **peopleSemSegNet**) with below files:

- **apps**: sample application for detection models and segmentation models
- **configs**: DeepStream nvinfer configure file and label files
- **post_processor**: include inference postprocessor for the models
- **graphs**: DeepStream sample graphs based on the Graph Composer tools.
- **models**:  To download sample models that are trained by  trained by [NVIDIA Train, Adapt, and Optimize(TAO) Toolkit SDK](https://developer.nvidia.com/tao-toolkit) run  `wget https://nvidia.box.com/shared/static/i1cer4s3ox4v8svbfkuj5js8yqm3yazo.zip -O models.zip`
- **TRT-OSS**: TRT(TensorRT) OSS libs for some platforms/systems (refer to the README to build lib for other platforms/systems)

The pipeline of the sample:

```
                                                                                                |---> encode --->filesink (save the output in local dir)  
H264/JPEG-->decoder-->tee -->| -- (batch size) -->|-->streammux--> nvinfer(detection)-->nvosd -->        
                                                                                                |---> display
```

## Prerequisites

* [Deepstream SDK 5.x onwards](https://developer.nvidia.com/deepstream-sdk)

   Make sure deepstream-test1 sample can run successful to verify your installation

   Note, For Unet/peopleSemSegNet model, you must use DS 5.1 onwards or later release.

* [TensorRT OSS (release/7.x branch)](https://github.com/NVIDIA/TensorRT/tree/release/7.0)

  This is **ONLY** needed when running *SSD*, *DSSD*, *RetinaNet*, *YOLOV3* , *YOLOV4* and *PeopleSegNet* models because some TRT plugins such as BatchTilePlugin required by these models is not supported by TensorRT7.x native package.

## Download

### 1. Download Source Code with SSH or HTTPS

```
// SSH
git clone -b release/tao3.0 git@github.com:NVIDIA-AI-IOT/deepstream_tao_apps.git
// or HTTPS
git clone -b release/tao3.0 https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
### 2. Download Models
Run below script to download models except multi_task model.

```
./download_models.sh
```

For multi_task, refer to https://docs.nvidia.com/tao/tao-toolkit/text/multitask_image_classification.html to train it by yourself

## Build

### 1. Build TRT OSS Plugin

Refer to below README to update libnvinfer_plugin.so* if want to run *SSD*, *DSSD*, *RetinaNet*, *PeopleSegNet*.

Note:This is also needed for *YOLOV3* , *YOLOV4* if you are using TRT version(such as TRT7.1) before TRT7.2

```
TRT-OSS/Jetson/README.md              // for Jetson platform
TRT-OSS/x86/README.md                 // for x86 platform
```

### 2. Build Sample Application

```
export CUDA_VER=xy.z                                      // xy.z is CUDA version, e.g. 10.2
make
```
## Run

```
Usage: ds-tao-detection -c pgie_config_file -i <H264 or JPEG filename> [-b BATCH] [-d]
    -h: print help info
    -c: pgie config file, e.g. pgie_frcnn_tao_config.txt
    -i: H264 or JPEG input file
    -b: batch size, this will override the value of "batch-size" in pgie config file
    -d: enable display, otherwise dump to output H264 or JPEG file
 
 e.g.
 ./apps/tao_segmentation/ds-tao-segmentation -c configs/unet_tao/pgie_unet_tao_config.txt -i $DS_SRC_PATH/samples/streams/sample_720p.h264
 ./apps/tao_classifier/ds-tao-classifier -c configs/multi_task_tao/pgie_multi_task_tao_config.txt -i $DS_SRC_PATH/samples/streams/sample_720p.h264
 [SHOW_MASK=1] ./apps/tao_detection/ds-tao-detection  -c configs/frcnn_tao/pgie_frcnn_tao_config.txt -i $DS_SRC_PATH/samples/streams/sample_720p.h264

 note:for PeopleSegNet, you need to set SHOW_MASK=1 if you need to display the instance mask
```

## Information for Customization

If you want to do some customization, such as training your own TAO model, running the model in other DeepStream pipeline, you should read below sections.
### TAO Models

To download the sample models that we have trained with [NVIDIA TAO Toolkit SDK](https://developer.nvidia.com/tao-toolkit) , run `wget https://nvidia.box.com/shared/static/i1cer4s3ox4v8svbfkuj5js8yqm3yazo.zip -O models.zip`

Refer [TAO Doc](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) for how to train the models, after training finishes, run `tao-export` to generate an `.etlt` model. This .etlt model can be deployed into DeepStream for fast inference as this sample shows.
This DeepStream sample app also supports the TensorRT engine(plan) file generated by running the `tao-converter` tool on the `.etlt` model.
The TensorRT engine file is hardware dependent, while the `.etlt` model is not. You may specify either a TensorRT engine file or a `.etlt` model in the DeepStream configuration file.

Note, for Unet/peopleSemSegNet/yolov3/yolov4 model, you must convert the etlt model to TensorRT engine file using `tao-convert` like following if you are using DS5.x:

```
tao-converter -e models/unet/unet_resnet18.etlt_b1_gpu0_fp16.engine -p input_1,1x3x608x960,1x3x608x960,1x3x608x960 -t fp16 -k tlt_encode -m 1 tlt_encode models/unet/unet_resnet18.etlt
```
### Label Files

The label file includes the list of class names for a model, which content varies for different models.  
User can find the detailed label information for the MODEL in the README.md and the label file under *configs/$(MODEL)_tao/*, e.g. ssd label informantion under *configs/ssd_tao/*
  
Note, for some models like FasterRCNN, DON'T forget to include "background" lable and change num-detected-classes in pgie configure file accordingly 

### DeepStream configuration file

The DeepStream configuration file includes some runtime parameters for DeepStream **nvinfer** plugin, such as model path, label file path, TensorRT inference precision, input and output node names, input dimensions and so on.  
In this sample, each model has its own DeepStream configuration file, e.g. pgie_dssd_tao_config.txt for DSSD model.
Please refer to [DeepStream Development Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_config.3.2.html) for detailed explanations of those parameters.

### Model Outputs

#### 1~2. Yolov3 / YoloV4

The model has the following four outputs:

- **num_detections**: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
- **nmsed_boxes**: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
- **nmsed_scores**: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
- **nmsed_classes**: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes

#### 3~6. RetinaNet / DSSD / SSD/ FasterRCNN

These three models have the same output layer named NMS which implementation can refer to TRT OSS [nmsPlugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin/nmsPlugin):

* an output of shape [batchSize, 1, keepTopK, 7] which contains nmsed box class IDs(1 value), nmsed box scores(1 value) and nmsed box locations(4 value)
* another output of shape [batchSize, 1, 1, 1] which contains the output nmsed box count.

#### 7. PeopleSegNet

The model has the following two outputs:

- **generate_detections**: A [batchSize, keepTopK, C*6] tensor containing the bounding box, class id, score
- **mask_head/mask_fcn_logits/BiasAdd**:  A [batchSize, keepTopK, C+1, 28*28] tensor containing the masks

#### 8~9. UNET/PeopleSemSegNet

- **softmax_1**: A [batchSize, H, W, C] tensor containing the scores for each class

#### 10. multi_task
- refer detailed [README](./configs/multi_task_tao/README.md) for how to configure and run the model
### TRT Plugins Requirements

>- **FasterRCNN**: cropAndResizePlugin,  proposalPlugin
>- **SSD/DSSD/RetinaNet**:  batchTilePlugin, nmsPlugin
>- **YOLOV3/YOLOV4**:  batchTilePlugin, resizeNearestPlugin, batchedNMSPlugin
>- **PeopleSegNet**:  generateDetectionPlugin, MultilevelCropAndResize, MultilevelProposeROI

## FAQ

### Measure The Inference Perf

```CQL
# 1.  Build TensorRT Engine through this smample, for example, build YoloV3 with batch_size=2
./ds-tao -c pgie_yolov3_tao_config.txt -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264 -b 2
## after this is done, it will generate the TRT engine file under models/$(MODEL), e.g. models/yolov3/ for above command.
# 2. Measure the Inference Perf with trtexec, following above example
cd models/yolov3/
trtexec --batch=2 --useSpinWait --loadEngine=yolo_resnet18.etlt_b2_gpu0_fp16.engine
## then you can find the per *BATCH* inference time in the trtexec output log
```

### About misc folder

```CQL
# The files in the folder are used by TAO 3.0 dev blogs:
## 1.  Training State-Of-The-Art Models for Classification and Object Detection with NVIDIA TAO Toolkit
## 2.  Real time vehicle license plate detection and recognition using NVIDIA TAO Toolkit
```
## Others Models

There are some special models which are not exactly detector, classifier or segmetation. The sample application of these special models are put in apps/tao_others. These samples should run on DeepStream 6.0 or above versions. Please refer to apps/tao_others/README.md document for details.

## Graph Composer Samples

Some special models needs special deepstream pipeline for running. The deepstream sample graphs for them are put in graphs/tao_others. Please refer to graphs/README.md file for more details.

## Known issues

