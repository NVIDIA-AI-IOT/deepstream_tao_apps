## Description
This sample is to show the following TAO models runing with DeepStream

* Retail Object Detection Binary
* Retail Object Recognition
* PeopleNet Transformer
* ReIdentificationNet
* PoseClassificationNet
* OCDNet
* OCRNet
* LPDNet
* LPRNet

## Prerequisition

* [DeepStream SDK 7.1 GA or above](https://developer.nvidia.com/deepstream-sdk)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [TAO models](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html)

  Nvidia has provides all the trainable models in NGC.
  The models used in the sample application are pre-trained models provided by TAO:

| Model name | NGC link  | Version |
|------------|-----------|---------|
| Retail Object Detection Binary | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection)|deployable_retail_object_detection_binary_v2.2.2.3|
|PeopleNet Transformer | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer)|deployable_v1.1|
|Retail Object Recognition|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition)|deployable_v2.0|
|ReIdentificationNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet)|deployable_v1.2|
|PoseClassificationNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/poseclassificationnet)|deployable_onnx_v1.0|
|OCDNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocdnet)|deployable_onnx_v2.4|
|OCRNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocrnet)|deployable_v2.1.1|
|LPDNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet)|pruned_v2.3.1|
|LPRNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet)|deployable_onnx_v1.1|

 ## Triton Server Settings
 
 The DeepStream sample applications can work as Triton client with the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), one of the following two methods can be used to set up the Triton Inference Server before starting a gst-nvinferserver inferncing DeepStream application.

 - Native Triton Inference Server, please refer to [Triton Server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server.md)
 - Stand-alone Triton Inference server, please refer to [Triton grpc server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server_grpc.md).
 
Please enable Triton or Triton gRPC inferencing with the app YAML configurations.

## Download

1. Download Project with HTTPS
```
    sudo apt update
    sudo apt install git-lfs
    git lfs install --skip-repo
    git clone -b master https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
2. Prepare Models and TensorRT engine

There are pre-trained TAO models available in [NGC](https://ngc.nvidia.com/catalog/models) for faciallandmarks, emotion, gesture, gaze and heart rate.

Please run the following script to download pre-trained models and generate GazeNet and Gesture engines with tao-converter tool:

```
    cd deepstream_tao_apps
    chmod 755 download_models.sh
    ./download_models.sh
```

## Build and Run
Build the application
```
    make
    cd apps/tao_others
```

Start to run the nvocdr application
Please prepare the nvocdr libs first, you can refer to [NVOCDR_README](./deepstream-nvocdr-app/README.md)
```
    cd deepstream-nvocdr-app
    ./deepstream-nvocdr-app <app YAML config file>
```

Start to run the pose classification application
```
    cd deepstream-pose-classification
    ./deepstream-pose-classification-app <app YAML config file>
```

Start to run the mdx perception application
```
    cd deepstream-mdx-perception-app
    ./deepstream-mdx-perception-app -c <txt config file>
OR
    ./deepstream-mdx-perception-app <app YAML config file>
```

Start to run the car license plate recognition sample application
```
    cd deepstream_lpr_app
    ##For US car plate recognition
    cp dict_us.txt dict.txt
    ##For Chinese car plate recognition
    cp dict_ch.txt dict.txt
    ##Run the sample app
    ./deepstream-lpr-app/deepstream-lpr-app <1:US car plate model|2: Chinese car plate model> \
         <1: output as h264 file| 2:fakesink 3:display output> <0:ROI disable|1:ROI enable> <infer|triton|tritongrpc> \
         <input mp4 file name> ... <input mp4 file name> <output file name>
OR
    ./deepstream-lpr-app/deepstream-lpr-app <app YAML config file>
```

A sample of mdx perception:

`./deepstream-mdx-perception-app  -c ../../../configs/app/peoplenet_reidentification.yml`

A sample of nvocdr:
Please prepare the nvocdr libs first, you can refer to [NVOCDR_README](./deepstream-nvocdr-app/README.md)

`./deepstream-nvocdr-app nvocdr_app_config.yml`

A sample of pose classification:

`./deepstream-pose-classification-app ../../../configs/app/deepstream_pose_classification_config.yaml`

A sample of the car license plate recognition:

`./deepstream-lpr-app/deepstream-lpr-app 1 1 1 infer /opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4 out`

or
`./deepstream-lpr-app/deepstream-lpr-app ../../../configs/app/lpr_app_infer_us_config.yml`

## Known Issue
1.For the deepstream-nvocdr-app, if the DeepStream version is lower than 6.4, the TensorRT OSS plugin is needed. Please refer to [NVOCDR_README](./deepstream-nvocdr-app/README.md). To avoid affecting the results of other apps, please replace the TensorRT plugin with the original one after running this app.
