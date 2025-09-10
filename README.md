# Integrate TAO model with DeepStream SDK

- [Integrate TAO model with DeepStream SDK](#integrate-tao-model-with-deepstream-sdk)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Download](#download)
    - [1. Download Source Code with SSH or HTTPS](#1-download-source-code-with-ssh-or-https)
    - [2. Download Models](#2-download-models)
  - [Triton Inference Server](#triton-inference-server)
  - [Build](#build)
    - [Build Sample Application](#build-sample-application)
  - [Run](#run)
  - [Information for Customization](#information-for-customization)
    - [TAO Models](#tao-models)
    - [Label Files](#label-files)
    - [DeepStream configuration file](#deepstream-configuration-file)
    - [Model Outputs](#model-outputs)
      - [1. CitySemSegFormer](#1-citysemsegformer)
      - [2. PoseClassification](#2-poseclassification)
      - [3~4. Retail Object Detection / PeopleNet Transformer](#34-retail-object-detection--peoplenet-transformer)
      - [5~6. Re-Identification / Retail Item Recognition](#56-re-identification--retail-item-recognition)
      - [7. BodyPose3DNet](#7-bodypose3dnet)
      - [8~9. OCDNet / OCRNet](#89-ocdnet--ocrnet)
      - [10~11. LPDNet / LPRNet](#1011-lpdnet--lprnet)
      - [12. Mask2Former](#12-mask2former)
  - [FAQ](#faq)
    - [Measure The Inference Perf](#measure-the-inference-perf)
    - [About misc folder](#about-misc-folder)
  - [Others Models](#others-models)
  - [TAO pyservicemaker samples](#tao-pyservicemaker-samples)
  - [Known issues](#known-issues)

## Description

This repository provides a DeepStream sample application based on [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) to run twelve TAO models ( **Peoplenet Transformer** / **CitySemSegformer** / **Mask2Former** /**Reidentification** / **Bodypose3dnet** / **Pose Classifier** / **OCDNet**/ **OCRNet**/ **LPDNet** / **LPRNet** / **Retail Detector**/ **Retail Object Recognition** ) with below files:

- **apps**: sample application for detection models and segmentation models
- **configs**: DeepStream nvinfer configure file and label files
- **post_processor**: include inference postprocessor for the models
- **models**: The models which will be used as samples.
- **TRT-OSS**: The OSS nvinfer plugin build and download instructions. The OSS plugins are needed for some models with DeepStream 7.1 GA.

The pipeline of the sample:

```
                                                                           |-->filesink(save the output in local dir)
                                                            |--> encode -->
                                                                           |-->fakesink(use -f option)
uridecoderbin -->streammux-->nvinfer(detection)-->nvosd-->
                                                            |--> display
```

## Prerequisites

* [DeepStream SDK 8.0 GA](https://developer.nvidia.com/deepstream-sdk)

   Make sure deepstream-test1 sample can run successful to verify your installation. According to the
   [document](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html),
   please run below command to install additional audio video packages.

  ```
  /opt/nvidia/deepstream/deepstream/user_additional_install.sh
  ```
* Eigen development packages
  ```
    sudo apt install libeigen3-dev
    cd /usr/include
    sudo ln -sf eigen3/Eigen Eigen
  ```

## Download

### 1. Download Source Code with SSH or HTTPS

```
sudo apt update
sudo apt install git-lfs
git lfs install --skip-repo
// SSH
git clone -b tao_ds8.0ga git@github.com:NVIDIA-AI-IOT/deepstream_tao_apps.git
// or HTTPS
git clone -b tao_ds8.0ga https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
//download submoudle
git submodule update --init --recursive
```
### 2. Download Models
Run below script to download models except multi_task and YoloV5 models.

```
sudo ./download_models.sh  # (sudo not required in case of docker containers)
```

### 3. Download Pre-built TensorRT OSS nvinfer plugin library

Please download the TensorRT OSS plugin according to your platforms

[x86 platform TRT OSS plugin download instruction](TRT-OSS/x86/README.md)

[Jetson platform TRT OSS plugin download instruction](TRT-OSS/Jetson/README.md)

## Triton Inference Server

The sample provides three inferencing methods. For the TensorRT based gst-nvinfer inferencing, please skip this part.

The DeepStream sample application can work as Triton client with the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), one of the following two methods can be used to set up the Triton Inference Server before starting a gst-nvinferserver inferncing DeepStream application.

 - Native Triton Inference Server, please refer to [Triton Server](triton_server.md)
 - Stand-alone Triton Inference server, please refer to [Triton grpc server](triton_server_grpc.md)

For the TAO sample applications, please enable Triton or Triton gRPC inferencing with the app YAML configurations.

E.G. With apps/tao_detection/ds-tao-detection, the "primary-gie" part in configs/app/det_app_config.yml can be modified as following:

```
primary-gie:
  #0:nvinfer, 1:nvinfeserver
  plugin-type: 1

  #peoplenet-transformer
  #config-file-path: ../nvinfer/peoplenet_transformer_tao/pgie_peoplenet_transformer_tao_config.yml
  config-file-path: ../triton/peoplenet_transformer_tao/pgie_peoplenet_transformer_tao_config.yml
  #config-file-path: ../triton-grpc/peoplenet_transformer_tao/pgie_peoplenet_transformer_tao_config.yml

```
And then run the app with the command:

```
./apps/tao_detection/ds-tao-detection configs/app/det_app_config.yml
```

## Build

### Build Sample Application

```
export CUDA_MODULE_LOADING=LAZY
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
  ./apps/tao_detection/ds-tao-detection configs/app/det_app_config.yml


note: If you want use multi-source, you can input multi -i input(e.g., -i uri -i uri...) 
      Only YAML configurations support Triton and Triton gRPC inferencing.
```
For detailed model information, pleasing refer to the following table:

note:  
  The default $DS_SRC_PATH is /opt/nvidia/deepstream/deepstream

|Model Type|Tao Model|Demo|
|-----------|----------|----|
|detector|peoplenet_transformer, retail_detector_binary|./apps/tao_detection/ds-tao-detection -c configs/nvinfer/peoplenet_transformer_tao/pgie_peoplenet_transformer_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_detection/ds-tao-detection configs/app/det_app_config.yml|
|segmentation|citySemSegFormer|./apps/tao_segmentation/ds-tao-segmentation -c configs/nvinfer/citysemsegformer_tao/pgie_citysemsegformer_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4 -w 960 -e 544<br>or<br>./apps/tao_segmentation/ds-tao-segmentation configs/app/seg_app_config.yml|
|instance segmentation|Mask2Former|export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection -c configs/nvinfer/mask2former_tao/pgie_mask2former_tao_config.yml -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection configs/app/ins_seg_app.yml|
|others|Re-identification, Retail Object Recognition, PoseClassificationNet, OCDNet, OCRNet, LPDNet, LPRNet, BodyPose3DNet|refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model|

Building the TensorRT engine of citySemSegFormer consumes a lot of device memory. Please `export CUDA_MODULE_LOADING=LAZY` to reduce device memory consumption. Please read [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) for details.

## Information for Customization

If you want to do some customization, such as training your own TAO models, running the model in other DeepStream pipeline, you should read below sections.
### TAO Models

Refer [TAO Doc](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) for how to train the models, after training finishes, run `tao-export` to generate an ONNX model. This ONNX model can be deployed into DeepStream for fast inference as this sample shows.
This DeepStream sample app also supports the TensorRT engine(plan) file generated by running the `trtexec` tool on the ONNX model.
The TensorRT engine file is hardware dependent, while the ONNX model is not. You may specify either a TensorRT engine file or a ONNX model in the DeepStream configuration file.
### Label Files

The label file includes the list of class names for a model, which content varies for different models.  
User can find the detailed label information for the MODEL in the README.md and the label file under *configs/$(MODEL)_tao/*, e.g. ssd label informantion under *configs/peoplenet_transformer_tao/*

### DeepStream configuration file

The DeepStream configuration file includes some runtime parameters for DeepStream **nvinfer** plugin or **nvinferserver** plugin, such as model path, label file path, TensorRT inference precision, input and output node names, input dimensions and so on.  
In this sample, each model has its own DeepStream configuration file, e.g. pgie_peoplenet_transformer_tao_config.txt for peoplenet transformer model.
Please refer to [DeepStream Development Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications) for detailed explanations of those parameters.

### Model Outputs

#### 1. CitySemSegFormer

- **argmax_1/output**: A [batchSize, H, W, 1] tensor containing the class id per pixel location

#### 2. PoseClassification

- refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model

#### 3~4. Retail Object Detection / PeopleNet Transformer

The model has the following two outputs:

- **pred_logits**: This is a [batch_size, num_queries, num_classes] tensor of data type float32. The
tensor contains probability values of each class.
- **pred_boxes**: This is a [batch_size, num_queries, 4] tensor of data type float32. The tensor
represents the 2D bounding box coordinates in the format of [center_x, center_y, width, height].

#### 5~6. Re-Identification / Retail Item Recognition

These models are trained to extract the embedding vector from an image. The image is the cropped area of a
bounding box from a primary-gie task, like people detection by `PeopleNet Transformer` or retail item detection
by `Retail Object Detection`. These embedding extraction models are typically arranged
as the secondary GIE module in a Deepstream pipeline.

##### Re-Identification uses ResNet50 backbone.
The output layer is:
- **fc_pred**: This is a [batch_size, embedding_size] tensor of data type float32. The tensor
contains the embedding vector of size `embedding_size = 256`.

##### Retail Item Recognition uses ResNet101 backbone.
The output layer is:
- **outputs**: This is a [batch_size, 2048] tensor of data type float32. The tensor contains the embedding
vector of size `2048`.

#### 7. BodyPose3DNet
The BodyPose3DNet models are used for 3D human pose estimation network, which aims to predict the skeleton for every person in a given input image which consists of keypoints and the connections between them.

The model's output layers:
- **pose2d**: The two dimension coordinates and probabilities of the keypoints which are in the [batch_size, keypoint_num, 3] tensor.
- **pose2d_org_img**: The two dimension coordinates and probabilities of the keypoints in the original image which are in the [batch_size, keypoint_num, 3] tensor.
- **pose25d**: The 2.5 dimension coordinates and probabilities of the keypoints in the original image which are in the [batch_size, keypoint_num, 4] tensor.
- **pose3d**: The 3 dimension coordinates of the keypoints which are in the [batch_size, keypoint_num, 3] tensor.

#### 8~9. OCDNet / OCRNet

##### OCDNet output layer
- **pred**:This is a [batchSize, H, W , 1] float tensor containing the probability of a pixel belongs to text
##### OCRNet output layer
- **output_prob**:This is a [batchSize, W/4] float tensor containing the confidence of each character in the text
- **output_id**:This is a [batchSize, W/4] integer tensor containing each character index of the text. This index can be mapped to character through the OCRNet charater list

#### 10~11. LPDNet / LPRNet

##### LPDNet output
- Category labels (lpd) and bounding-box coordinates for each detected license plate in the input image.

##### LPRNet output
- characters id sequence. (DeepStream post-process plugin is needed to get the final license plate)

#### 12. Mask2Former
The model has the following three outputs:

- **pred_classes**: A [100] tensor containing the prediction classes
- **pred_masks**:  A [100x800x800] tensor containing the prediction masks
- **pred_scores**:  A [100] tensor containing the prediction scores
## FAQ

### Measure The Inference Perf

```CQL
# 1.  Build TensorRT Engine through this smample, for example, build YoloV3 with batch_size=2
./apps/tao_detection/ds-tao-detection -c configs/nvinfer/peoplenet_transformer_tao/pgie_peoplenet_transformer_tao_config.txt -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 -b 2
## after this is done, it will generate the TRT engine file under models/$(MODEL), e.g. models/peoplenet_transformer/ for above command.
# 2. Measure the Inference Perf with trtexec, following above example
cd models/peoplenet_transformer/
/usr/src/tensorrt/bin/trtexec --useSpinWait --loadEngine=resnet50_peoplenet_transformer_op17.onnx_b2_gpu0_fp16.engine
## then you can find the per *BATCH* inference time in the trtexec output log
```

### About misc folder

```CQL
# The files in the folder are used by TAO dev blogs:
## 1.  Training State-Of-The-Art Models for Classification and Object Detection with NVIDIA TAO Toolkit
## 2.  Real time vehicle license plate detection and recognition using NVIDIA TAO Toolkit
```
## Others Models

There are some special models which are not exactly detector, classifier or segmetation. The sample application of these special models are put in apps/tao_others. These samples should run on DeepStream 6.1 or above versions. Please refer to apps/tao_others/README.md document for details.

## TAO pyservicemaker samples

PyServiceMaker is a python framework that aims to streamline the development process for Deepstream applications. It provides a convenient and efficient means to rapidly construct application prototypes, enabling developers to expedite the prototyping phase and focus on the core functionality of their projects.

Please refer to [REAME](pysm-apps/README.md) file for more details.

## Known issues

1. Currently the citySemSegFormer model only supports batch-size 1.
2. If the segmentation results can't overlay the entire frame, please set SEG_OUTPUT_WIDTH/SEG_OUTPUT_HEIGHT to the model's width/height.
