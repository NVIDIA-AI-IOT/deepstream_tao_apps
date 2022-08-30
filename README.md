# Integrate TAO model with DeepStream SDK

- [Integrate TAO model with DeepStream SDK](#integrate-tao-model-with-deepstream-sdk)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Download](#download)
    - [1. Download Source Code with SSH or HTTPS](#1-download-source-code-with-ssh-or-https)
    - [2. Download Models](#2-download-models)
  - [Build](#build)
    - [Build Sample Application](#build-sample-application)
  - [Run](#run)
  - [Information for Customization](#information-for-customization)
    - [TAO Models](#tao-models)
    - [Label Files](#label-files)
    - [DeepStream configuration file](#deepstream-configuration-file)
    - [Model Outputs](#model-outputs)
      - [1~3. Yolov3 / YoloV4 / Yolov4-tiny / Yolov5](#13-yolov3-yolov4-yolov4-tiny-yolov5)
      - [4~7. RetinaNet / DSSD / SSD/ FasterRCNN](#47-retinanet-dssd-ssd-fasterrcnn)
      - [8. PeopleSegNet](#8-peoplesegnet)
      - [9~10. UNET/PeopleSemSegNet](#910-unetpeoplesemsegnet)
      - [11. multi_task](#11-multitask)
      - [12. EfficientDet](#12-efficientdet)
      - [13. FaceDetect / Facial Landmarks Estimation / EmotionNet / Gaze Estimation / GestureNet / HeartRateNet / BodyPoseNet](#13-facedetect-facial-landmarks-estimation-emotionnet-gaze-estimation-gesturenet-heartratenet-bodyposenet)
    - [Calibration file with TensorRT version](#calibration-file-with-tensorrt-version)
  - [FAQ](#faq)
    - [Measure The Inference Perf](#measure-the-inference-perf)
    - [About misc folder](#about-misc-folder)
  - [Others Models](#others-models)
  - [Graph Composer Samples](#graph-composer-samples)
  - [Known issues](#known-issues)

## Description

This repository provides a DeepStream sample application based on [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) to run eleven TAO models (**Faster-RCNN** / **YoloV3** / **YoloV4** / **YoloV5** /**SSD** / **DSSD** / **RetinaNet**/ **PeopleSegNet**/ **UNET**/ **multi_task**/ **peopleSemSegNet**) with below files:

- **apps**: sample application for detection models and segmentation models
- **configs**: DeepStream nvinfer configure file and label files
- **post_processor**: include inference postprocessor for the models
- **graphs**: DeepStream sample graphs based on the Graph Composer tools.
- **models**: The models which will be used as samples.
- **TRT-OSS**: The OSS nvinfer plugin build and download instructions. The OSS plugins are not needed for DeepStream 6.1.1 GA.

The pipeline of the sample:

```
                                                                           |-->filesink(save the output in local dir)
                                                            |--> encode -->
                                                                           |-->fakesink(use -f option)
uridecoderbin -->streammux-->nvinfer(detection)-->nvosd-->
                                                            |--> display
```

## Prerequisites

* [DeepStream SDK 6.1.1 GA](https://developer.nvidia.com/deepstream-sdk)

   Make sure deepstream-test1 sample can run successful to verify your installation

## Download

### 1. Download Source Code with SSH or HTTPS

```
// SSH
git clone -b release/tao3.0_ds6.1.1ga git@github.com:NVIDIA-AI-IOT/deepstream_tao_apps.git
// or HTTPS
git clone release/tao3.0_ds6.1.1ga https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
### 2. Download Models
Run below script to download models except multi_task and YoloV5 models.

```
./download_models.sh
```

For multi_task, refer to https://docs.nvidia.com/tao/tao-toolkit/text/multitask_image_classification.html to train and generate the model.

For yolov5, refer to [yolov5_gpu_optimization](https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization) to generate the onnx model

## Build

### Build Sample Application

```
export CUDA_VER=xy.z                                      // xy.z is CUDA version, e.g. 10.2
make
```
## Run

```

1.Usage: ds-tao-detection -c pgie_config_file -i <H264 or JPEG file uri> [-b BATCH] [-d] [-f] [-l]
    -h: print help info
    -c: pgie config file, e.g. pgie_frcnn_tao_config.txt
    -i: uri of the input file, start with the file:///, e.g. file:///.../video.mp4
    -b: batch size, this will override the value of "batch-size" in pgie config file
    -d: enable display, otherwise it will dump to output MP4 or JPEG file without -f option
    -f: use fakesink mode
    -l: use loop mode

2.Usage: ds-tao-detection <yaml file uri>
  e.g.
  ./apps/tao_detection/ds-tao-detection configs/app/det_app_frcnn.yml


note: If you want use multi-source, you can input multi -i input(e.g., -i uri -i uri...) 
```
For detailed model information, pleasing refer to the following table:

note:  
  The default $DS_SRC_PATH is /opt/nvidia/deepstream/deepstream

|Model Type|Tao Model|Demo|
|-----------|----------|----|
|detector|dssd, efficientdet, frcnn, retinanet, ssd, yolov3, yolov4-tiny, yolov4, yolov5|./apps/tao_detection/ds-tao-detection -c configs/dssd_tao/pgie_dssd_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_detection/ds-tao-detection configs/app/det_app_frcnn.yml|
|classifier|multi-task|./apps/tao_classifier/ds-tao-classifier -c configs/multi_task_tao/pgie_multi_task_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_classifier/ds-tao-classifier configs/app/multi_task_app_config.yml|
|segmentation|peopleSemSegNet, unet|./apps/tao_segmentation/ds-tao-segmentation -c configs/peopleSemSegNet_tao/pgie_peopleSemSegNet_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_segmentation/ds-tao-segmentation configs/app/seg_app_unet.yml|
|instance segmentation|peopleSegNet|export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection -c configs/peopleSegNet_tao/pgie_peopleSegNet_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection configs/app/ins_seg_app_peopleSegNet.yml|
|others|FaceDetect, Facial Landmarks Estimation, EmotionNet, Gaze Estimation, GestureNet, HeartRateNet, BodyPoseNet|refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model|

## Information for Customization

If you want to do some customization, such as training your own TAO model, running the model in other DeepStream pipeline, you should read below sections.
### TAO Models

To download the sample models that we have trained with [NVIDIA TAO Toolkit SDK](https://developer.nvidia.com/tao-toolkit) , run `wget https://nvidia.box.com/shared/static/vynsy1tzhdeiwt7a5j44ssitqlm2a9rg -O models.zip`

Refer [TAO Doc](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) for how to train the models, after training finishes, run `tao-export` to generate an `.etlt` model. This .etlt model can be deployed into DeepStream for fast inference as this sample shows.
This DeepStream sample app also supports the TensorRT engine(plan) file generated by running the `tao-converter` tool on the `.etlt` model.
The TensorRT engine file is hardware dependent, while the `.etlt` model is not. You may specify either a TensorRT engine file or a `.etlt` model in the DeepStream configuration file.

Note, for Unet/peopleSemSegNet/yolov3/yolov4/yolov5 model, you can also convert the etlt model to TensorRT engine file using `tao-converter` like following:

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

#### 1~3. Yolov3 / YoloV4 / Yolov4-tiny / Yolov5

The model has the following four outputs:

- **num_detections**: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
- **nmsed_boxes**: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
- **nmsed_scores**: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
- **nmsed_classes**: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes

#### 4~7. RetinaNet / DSSD / SSD/ FasterRCNN

These three models have the same output layer named NMS which implementation can refer to TRT OSS [nmsPlugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin/nmsPlugin):

* an output of shape [batchSize, 1, keepTopK, 7] which contains nmsed box class IDs(1 value), nmsed box scores(1 value) and nmsed box locations(4 value)
* another output of shape [batchSize, 1, 1, 1] which contains the output nmsed box count.

#### 8. PeopleSegNet

The model has the following two outputs:

- **generate_detections**: A [batchSize, keepTopK, C*6] tensor containing the bounding box, class id, score
- **mask_head/mask_fcn_logits/BiasAdd**:  A [batchSize, keepTopK, C+1, 28*28] tensor containing the masks

#### 9~10. UNET/PeopleSemSegNet

- **softmax_1**: A [batchSize, H, W, C] tensor containing the scores for each class

#### 11. multi_task
- refer detailed [README](./configs/multi_task_tao/README.md) for how to configure and run the model

#### 12. EfficientDet

The model has the following four outputs:

- **num_detections**: This is a [batch_size, 1] tensor of data type int32. The last dimension is a scalar indicating the number of valid detections per batch image. It can be less than max_output_boxes. Only the top num_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_classes[i] are valid.
- **detection_boxes**: This is a [batch_size, max_output_boxes, 4] tensor of data type float32 or float16, containing the coordinates of non-max suppressed boxes. The output coordinates will always be in BoxCorner format, regardless of the input code type.
- **detection_scores**: This is a [batch_size, max_output_boxes] tensor of data type float32 or float16, containing the scores for the boxes.
- **detection_classes**: This is a [batch_size, max_output_boxes] tensor of data type int32, containing the classes for the boxes.

#### 13. FaceDetect / Facial Landmarks Estimation / EmotionNet / Gaze Estimation / GestureNet / HeartRateNet / BodyPoseNet
- refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model

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

There are some special models which are not exactly detector, classifier or segmetation. The sample application of these special models are put in apps/tao_others. These samples should run on DeepStream 6.1 or above versions. Please refer to apps/tao_others/README.md document for details.

## Graph Composer Samples

Some special models needs special deepstream pipeline for running. The deepstream sample graphs for them are put in graphs/tao_others. Please refer to graphs/README.md file for more details.

## Known issues

For some yolo models, some layers of the models should use FP32 precision. This is a network characteristics that the accuracy drops rapidly when maximum layers are run in INT8 precision. Please refer the [layer-device-precision](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) for more details.
